"""Run with PYTHONPATH=toolchain python -m mfc.volume_filter config.json."""

import argparse
import copy
import hashlib
import json
import os
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

from .closures import closure_fields, reconstruct
from .maps import ParticleMap
from .numerics import Grid
from .reader import Reader

VERSION = 3
STATISTICS = ("mean", "variance", "skewness", "kurtosis")
KEY_FORMAT = "{field}.{statistic}.sigma{filter_index}"


class ParticleProgress:
    """Flush periodic particle counts to terminals and batch-job logs."""

    def __init__(self, label, interval):
        self.label, self.interval = label, interval
        self.start = self.last = perf_counter()

    def __call__(self, completed, total):
        now = perf_counter()
        if completed not in (0, total) and now - self.last < self.interval:
            return
        percent = 100 * completed / total if total else 100.0
        print(f"{self.label}: particles {completed}/{total} ({percent:.1f}%), elapsed {now - self.start:.1f}s", flush=True)
        self.last = now


def atomic_save(path, arrays):
    """Publish final output only after a complete write and flush."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as stream:
            # Stream NPZ entries, so derived means/variances do not all occupy RAM.
            entries = arrays.items() if isinstance(arrays, dict) else arrays
            with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
                for name, array in entries:
                    with archive.open(name + ".npy", "w", force_zip64=True) as entry:
                        np.lib.format.write_array(entry, np.asarray(array), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate(config):
    if config.get("periodic") != [True, True, True]:
        raise ValueError("This processor requires periodic=[true,true,true]")
    physics = config["physics"]
    if physics.get("eos") != "ideal_gas" or physics.get("transport") != "constant":
        raise ValueError("Only a single calorically perfect ideal gas with constant transport is supported")
    for name in ("gamma", "gas_constant", "mu", "conductivity"):
        if not np.isfinite(physics[name]):
            raise ValueError(f"Nonfinite physics.{name}")
    if physics["gamma"] <= 1 or physics["gas_constant"] <= 0 or min(physics["mu"], physics["conductivity"]) < 0:
        raise ValueError("Invalid gas/transport constants")
    if not 0 < config.get("alpha_min", 1e-6) < 1:
        raise ValueError("alpha_min must lie between zero and one")
    if config["surface"].get("thermal_bc") not in ("adiabatic", "reconstruct"):
        raise ValueError("surface.thermal_bc must be adiabatic or reconstruct")
    if config["surface"].get("velocity_bc", "no_slip") not in ("no_slip", "slip"):
        raise ValueError("surface.velocity_bc must be no_slip or slip")
    sigmas = np.asarray(config["sigmas"], dtype=float)
    if sigmas.ndim != 1 or not len(sigmas) or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0) or len(set(sigmas)) != len(sigmas):
        raise ValueError("sigmas must be distinct positive standard deviations")
    snapshots = config["snapshots"]
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("Provide at least one snapshot")
    steps = [item.get("step") if isinstance(item, dict) else item for item in snapshots]
    if any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("Snapshot steps must be nonnegative integers (MFC save indices)")
    if len(set(steps)) != len(steps):
        raise ValueError("Snapshot steps must be unique")
    # Normalize legacy objects before hashing; physical times are not inputs.
    config["snapshots"] = steps
    config.pop("time_weighting", None)
    config.pop("save_snapshot_maps", None)


def pack(moments, metadata):
    yield "metadata", np.array(json.dumps(metadata, sort_keys=True))
    for (name, sigma_index), value in moments.items():
        for statistic in STATISTICS:
            array = np.where(value.weight > 0, value.mean, np.nan) if statistic == "mean" else getattr(value, statistic)
            yield KEY_FORMAT.format(field=name, statistic=statistic, filter_index=sigma_index), array


def process_snapshots(config, root, indices, progress_interval, worker_id=None):
    """Process one snapshot at a time within one independent worker."""
    grid = Grid(**config["grid"])
    reader = Reader(config["io"], grid, root)
    mapper = ParticleMap(grid, config["map"])
    moments, records = {}, []
    selected = set(config.get("fields", []))
    for index in indices:
        step = config["snapshots"][index]
        snapshot_start = perf_counter()
        label = f"Snapshot {index + 1}/{len(config['snapshots'])} step={step}"
        if worker_id is not None:
            label = f"Worker {worker_id} {label}"

        def report(message):
            print(f"{label}: {message} [snapshot elapsed {perf_counter() - snapshot_start:.2f}s]", flush=True)

        report("reading particle state")
        fingerprint = reader.fingerprint(step)
        particles = reader.particles(step)
        time = float(particles[0, 0])  # File timestamp is descriptive metadata only.
        print(f"{label}: time={time}, particles={len(particles)}; reading/reconstructing fluid state", flush=True)
        state = reconstruct(grid, reader.fields(step, report=report), particles, config["physics"], report=report)
        for sigma_index, sigma in enumerate(config["sigmas"]):
            filter_label = f"{label} filter={sigma_index + 1}/{len(config['sigmas'])} sigma={sigma}"
            print(f"{filter_label}: computing full-domain closure fields and mapping particles", flush=True)
            seen = set()
            compute_start = perf_counter()
            for name, field in closure_fields(grid, state, particles, sigma, config):
                report(f"filter {sigma_index + 1} {name}: full-domain field ready in {perf_counter() - compute_start:.2f}s")
                seen.add(name)
                if selected and name not in selected:
                    compute_start = perf_counter()
                    continue
                particle = mapper.particles(field, particles, progress=ParticleProgress(f"{filter_label} {name}", progress_interval))
                key = (name, sigma_index)
                if key not in moments:
                    moments[key] = particle
                else:
                    moments[key].merge(particle)
                compute_start = perf_counter()
            if selected - seen:
                raise ValueError(f"Unknown fields: {sorted(selected - seen)}")
        del state
        if reader.fingerprint(step) != fingerprint:
            raise ValueError("Input changed while processing; only consume completed snapshots")
        records.append({"index": index, "inputs": fingerprint, "particle_count": len(particles), "time": time, "seconds": perf_counter() - snapshot_start})
        report("complete; statistics retained in memory")
    return moments, records


def run(config_path, progress_interval=5.0, snapshot_workers=1):
    if not np.isfinite(progress_interval) or progress_interval < 0:
        raise ValueError("progress_interval must be finite and nonnegative")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate(config)
    output = config_path.parent / config["output"]
    if output.exists():
        raise FileExistsError(f"{output} exists; select a new output path or remove the old file")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    metadata = {
        "version": VERSION,
        "output_mode": "final_only",
        "key_format": KEY_FORMAT,
        "config_hash": digest,
        "config": config,
        "statistics": {
            "ensemble": "particle_timestep_pairs",
            "pooled_weighting": "equal_realizations",
            "outputs": list(STATISTICS),
            "max_order": 4,
            "normalization": "population",
            "kurtosis": "Pearson",
        },
        "completed": 0,
        "inputs": [],
        "particle_counts": [],
        "snapshot_times": [],
    }
    if type(snapshot_workers) is not int or snapshot_workers < 1:
        raise ValueError("snapshot_workers must be a positive integer")
    workers = min(snapshot_workers, len(config["snapshots"]))
    worker_config = copy.deepcopy(config)
    budget = config.get("fft_workers", 1)
    if workers > 1:
        if type(budget) is not int or budget < workers:
            raise ValueError("fft_workers is the total CPU budget and must be >= snapshot_workers")
        worker_config["fft_workers"] = budget // workers
        from .parallel import run_workers

        print(f"Processing with {workers} snapshot workers, {worker_config['fft_workers']} FFT threads each", flush=True)
        moments, records = run_workers(worker_config, config_path.parent, workers, progress_interval)
    else:
        moments, records = process_snapshots(worker_config, config_path.parent, range(len(config["snapshots"])), progress_interval)
    records.sort(key=lambda item: item["index"])
    if [item["index"] for item in records] != list(range(len(config["snapshots"]))):
        raise RuntimeError("Snapshot workers did not cover each configured snapshot exactly once")
    metadata["completed"] = len(records)
    metadata["inputs"] = [item["inputs"] for item in records]
    metadata["particle_counts"] = [item["particle_count"] for item in records]
    metadata["snapshot_times"] = [item["time"] for item in records]
    metadata["snapshot_seconds"] = [item["seconds"] for item in records]
    metadata["execution"] = {"snapshot_workers": workers, "fft_workers_per_worker": worker_config.get("fft_workers", 1)}
    print(f"Writing final statistics: {output}", flush=True)
    atomic_save(output, pack(moments, metadata))
    print(f"Final output saved: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Seconds between particle progress lines (default: 5; 0 prints every particle)")
    parser.add_argument("--snapshot-workers", type=int, default=1, help="Independent snapshot processes on this node (default: 1); fft_workers is their total CPU budget")
    args = parser.parse_args()
    if not np.isfinite(args.progress_interval) or args.progress_interval < 0:
        parser.error("--progress-interval must be finite and nonnegative")
    if args.snapshot_workers < 1:
        parser.error("--snapshot-workers must be positive")
    run(args.config, progress_interval=args.progress_interval, snapshot_workers=args.snapshot_workers)


if __name__ == "__main__":
    main()
