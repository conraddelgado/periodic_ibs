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
from .diagnostics import capture, history_arrays, validate_diagnostics
from .maps import Moments, ParticleMap
from .numerics import Grid
from .reader import Reader

VERSION = 3
STATISTICS = ("mean", "variance", "skewness", "kurtosis")
KEY_FORMAT = "{field}.{statistic}.sigma{filter_index}"
PARTICLE_KEY_FORMAT = "{field}.{statistic}"
PARTICLE_FIELDS = {f"particle_{kind}_{axis}": start + offset for kind, start in (("force", 1), ("torque", 4)) for offset, axis in enumerate("xyz")}


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
    validate_diagnostics(config)


def pack(moments, metadata, records=None):
    yield "metadata", np.array(json.dumps(metadata, sort_keys=True))
    for (name, sigma_index), value in moments.items():
        for statistic in STATISTICS:
            array = np.where(value.weight > 0, value.mean, np.nan) if statistic == "mean" else getattr(value, statistic)
            key_format = PARTICLE_KEY_FORMAT if sigma_index is None else KEY_FORMAT
            yield key_format.format(field=name, statistic=statistic, filter_index=sigma_index), array

    if records is not None:
        yield from history_arrays(records)


def process_snapshots(config, root, indices, progress_interval, worker_id=None):
    """Process one snapshot at a time within one independent worker."""
    grid = Grid(**config["grid"])
    reader = Reader(config["io"], grid, root)
    mapper = ParticleMap(grid, config["map"], workers=config.get("fft_workers", 1))
    moments, records = {}, []
    selected = set(config.get("fields", []))
    diagnostic_options = config.get("diagnostics")
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
        history = {} if diagnostic_options is not None else None
        if history is not None and (not np.all(np.isfinite(particles[:, 0])) or np.any(particles[:, 0] != particles[0, 0])):
            raise ValueError("Diagnostic particle timestamps must be finite and identical within a snapshot")
        for name, column in PARTICLE_FIELDS.items():
            key = (name, None)
            if key not in moments:
                moments[key] = Moments(())
            snapshot_load = Moments(()) if history is not None and diagnostic_options["particle_loads"] else None
            for value in particles[:, column]:
                moments[key].update(value)
                if snapshot_load is not None:
                    snapshot_load.update(value)
            if snapshot_load is not None:
                history[name] = capture(snapshot_load)
        time = float(particles[0, 0])  # File timestamp is descriptive metadata only.
        state = reconstruct(grid, reader.fields(step), particles, config["physics"])
        report(f"fluid ready; time={time}, particles={len(particles)}, map backend={mapper.backend}")
        for sigma_index, sigma in enumerate(config["sigmas"]):
            filter_label = f"{label} filter={sigma_index + 1}/{len(config['sigmas'])} sigma={sigma}"
            print(f"{filter_label}: computing full-domain closure fields and mapping particles", flush=True)
            seen = set()
            compute_start = perf_counter()
            for name, field in closure_fields(grid, state, particles, sigma, config):
                compute_seconds = perf_counter() - compute_start
                seen.add(name)
                if selected and name not in selected:
                    compute_start = perf_counter()
                    continue
                map_start = perf_counter()
                particle = mapper.particles(field, particles)
                if history is not None and name in diagnostic_options["fields"]:
                    history[f"{name}.sigma{sigma_index}"] = capture(particle, diagnostic_options["map_indices"])
                key = (name, sigma_index)
                if key not in moments:
                    moments[key] = particle
                else:
                    moments[key].merge(particle)
                print(f"{filter_label} {name}: particles {len(particles)}/{len(particles)}; compute {compute_seconds:.2f}s, map/stats {perf_counter() - map_start:.2f}s", flush=True)
                compute_start = perf_counter()
            if selected - seen:
                raise ValueError(f"Unknown fields: {sorted(selected - seen)}")
            if diagnostic_options is not None and set(diagnostic_options["fields"]) - seen:
                raise ValueError(f"Unknown diagnostic fields: {sorted(set(diagnostic_options['fields']) - seen)}")
        reconstruction = state.reconstruction_metadata()
        del state
        if reader.fingerprint(step) != fingerprint:
            raise ValueError("Input changed while processing; only consume completed snapshots")
        record = {
            "index": index,
            "step": step,
            "inputs": fingerprint,
            "particle_count": len(particles),
            "time": time,
            "seconds": perf_counter() - snapshot_start,
            "map_backend": mapper.backend,
            "reconstruction": reconstruction,
        }
        if history is not None:
            record["diagnostics"] = history
        records.append(record)
        derivatives = reconstruction["derivatives"].values()
        gap_cells = sum(info["first_order_cells"] + info["affine_fit_cells"] for info in derivatives)
        surface = reconstruction["surface"]
        if gap_cells or any(surface[name] for name in ("buried_points", "reduced_offset_points", "affine_fit_points")):
            report(f"limited reconstruction: {gap_cells} cell/axis derivatives; surface {surface}")
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
    metadata["particle_statistics"] = {
        "fields": list(PARTICLE_FIELDS),
        "key_format": PARTICLE_KEY_FORMAT,
        "shape": [],
        "source": "ib_state",
        "record_columns_one_based": {name: column + 1 for name, column in PARTICLE_FIELDS.items()},
        "realizations": sum(metadata["particle_counts"]),
        "weighting": "equal_particle_timestep_pairs",
    }
    metadata["snapshot_seconds"] = [item["seconds"] for item in records]
    metadata["reconstruction"] = {
        "derivatives": "second_order_then_first_order_then_local_affine",
        "surface": "normal_extrapolation_then_reduced_offset_then_local_affine",
        "affine_radius_cells": 2,
        "rank_deficiency": "minimum_norm_gradient; unresolved components are zero",
        "overlaps": "exclude buried sphere quadrature without area renormalization",
        "snapshots": [item["reconstruction"] for item in records],
    }
    metadata["execution"] = {"snapshot_workers": workers, "fft_workers_per_worker": worker_config.get("fft_workers", 1), "map_backend": records[0]["map_backend"]}
    diagnostic_records = records if "diagnostics" in config else None
    if diagnostic_records is not None:
        metadata["diagnostics"] = {
            "version": 1,
            "prefix": "diagnostics.",
            "shape": ["snapshot", "probe"],
            "map_indices": config["diagnostics"]["map_indices"],
            "moments": ["count", "mean", "m2", "m3", "m4"],
            "central_sums": True,
            "particle_load_probes": 1,
            "weighting": "equal_particle_timestep_pairs",
        }
    print(f"Writing final statistics: {output}", flush=True)
    atomic_save(output, pack(moments, metadata, diagnostic_records))
    print(f"Final output saved: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Deprecated compatibility option; each selected field now prints one completion line")
    parser.add_argument("--snapshot-workers", type=int, default=1, help="Independent snapshot processes on this node (default: 1); fft_workers is their total CPU budget")
    args = parser.parse_args()
    if not np.isfinite(args.progress_interval) or args.progress_interval < 0:
        parser.error("--progress-interval must be finite and nonnegative")
    if args.snapshot_workers < 1:
        parser.error("--snapshot-workers must be positive")
    run(args.config, progress_interval=args.progress_interval, snapshot_workers=args.snapshot_workers)


if __name__ == "__main__":
    main()
