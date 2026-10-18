"""Run with PYTHONPATH=toolchain python -m mfc.volume_filter config.json."""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np

from .closures import closure_fields, reconstruct
from .maps import Moments, ParticleMap
from .numerics import Grid
from .reader import Reader

VERSION = 2


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
    """Replace a checkpoint only after a complete write and flush."""
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


def pack(moments, metadata):
    yield "metadata", np.array(json.dumps(metadata, sort_keys=True))
    for key, value in moments.items():
        yield key + ".count", value.count
        yield key + ".weight", value.weight
        yield key + ".mean", np.where(value.weight > 0, value.mean, np.nan)
        for order in (2, 3, 4):
            yield key + f".m{order}", getattr(value, f"m{order}")
            yield key + f".central_moment{order}", value.central_moment(order)
        for order in (1, 2, 3, 4):
            yield key + f".raw_moment{order}", value.raw_moment(order)
        yield key + ".variance", value.variance
        yield key + ".skewness", value.skewness
        yield key + ".kurtosis", value.kurtosis
        yield key + ".excess_kurtosis", value.kurtosis - 3


def restore(path, digest):
    moments = {}
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["version"] != VERSION:
            raise ValueError("Checkpoint version mismatch: moments 3-4 and realization counts require reprocessing the snapshots into a new output path")
        if metadata["config_hash"] != digest:
            raise ValueError("Checkpoint configuration mismatch; use a new output path")
        for key in data.files:
            if key.endswith(".weight"):
                name = key[:-7]
                result = Moments(data[key].shape)
                result.weight = data[key].copy()
                result.count = data[name + ".count"].copy()
                result.mean = np.where(result.weight > 0, data[name + ".mean"], 0)
                for order in (2, 3, 4):
                    setattr(result, f"m{order}", data[name + f".m{order}"].copy())
                moments[name] = result
    return moments, metadata


def run(config_path, resume=False, max_snapshots=None, progress_interval=5.0):
    if not np.isfinite(progress_interval) or progress_interval < 0:
        raise ValueError("progress_interval must be finite and nonnegative")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate(config)
    grid = Grid(**config["grid"])
    reader = Reader(config["io"], grid, config_path.parent)
    mapper = ParticleMap(grid, config["map"])
    output = config_path.parent / config["output"]
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    metadata = {
        "version": VERSION,
        "config_hash": digest,
        "config": config,
        "statistics": {
            "ensemble": "particle_timestep_pairs",
            "pooled_weighting": "equal_realizations",
            "snapshot_weighting": "equal_snapshots",
            "max_order": 4,
            "normalization": "population",
            "kurtosis": "Pearson",
        },
        "completed": 0,
        "inputs": [],
        "particle_counts": [],
        "snapshot_times": [],
    }
    moments = {}
    if output.exists():
        if not resume:
            raise FileExistsError(f"{output} exists; use --resume or select a new output")
        moments, metadata = restore(output, digest)
        for step, previous in zip(config["snapshots"], metadata["inputs"]):
            if json.loads(json.dumps(reader.fingerprint(step))) != previous:
                raise ValueError("A previously processed input file changed")
    elif resume:
        raise FileNotFoundError(f"Cannot resume missing checkpoint {output}")
    selected = set(config.get("fields", []))
    start = metadata["completed"]
    end = len(config["snapshots"]) if max_snapshots is None else min(len(config["snapshots"]), start + max_snapshots)
    for index in range(start, end):
        step = config["snapshots"][index]
        label = f"Snapshot {index + 1}/{len(config['snapshots'])} step={step}"
        print(f"{label}: reading particle state", flush=True)
        fingerprint = reader.fingerprint(step)
        particles = reader.particles(step)
        time = float(particles[0, 0])  # File timestamp is descriptive metadata only.
        print(f"{label}: time={time}, particles={len(particles)}; reading/reconstructing fluid state", flush=True)
        state = reconstruct(grid, reader.fields(step), particles, config["physics"])
        snapshot_maps = {}
        for sigma_index, sigma in enumerate(config["sigmas"]):
            filter_label = f"{label} filter={sigma_index + 1}/{len(config['sigmas'])} sigma={sigma}"
            print(f"{filter_label}: computing full-domain closure fields and mapping particles", flush=True)
            seen = set()
            for name, field in closure_fields(grid, state, particles, sigma, config):
                seen.add(name)
                if selected and name not in selected:
                    continue
                particle = mapper.particles(field, particles, progress=ParticleProgress(f"{filter_label} {name}", progress_interval))
                key = f"sigma{sigma_index}.{name}"
                valid_mean = np.where(particle.weight > 0, particle.mean, np.nan)
                # Keep time-series diagnostics distinct from the equally weighted
                # particle/timestep ensemble, even for irregular snapshot times.
                for suffix in (".time", ".pooled"):
                    if key + suffix not in moments:
                        moments[key + suffix] = Moments(mapper.shape)
                moments[key + ".time"].update(valid_mean)
                moments[key + ".pooled"].merge(particle)
                if config.get("save_snapshot_maps", False):
                    snapshot_maps[key + ".mean"] = valid_mean
                    snapshot_maps[key + ".particle_variance"] = particle.variance
                    snapshot_maps[key + ".particle_count"] = particle.count
                    snapshot_maps[key + ".particle_central_moment3"] = particle.central_moment(3)
                    snapshot_maps[key + ".particle_central_moment4"] = particle.central_moment(4)
                    snapshot_maps[key + ".particle_skewness"] = particle.skewness
                    snapshot_maps[key + ".particle_kurtosis"] = particle.kurtosis
            if selected - seen:
                raise ValueError(f"Unknown fields: {sorted(selected - seen)}")
        del state
        if reader.fingerprint(step) != fingerprint:
            raise ValueError("Input changed while processing; only consume completed snapshots")
        metadata["completed"] = index + 1
        metadata["inputs"].append(fingerprint)
        metadata["particle_counts"].append(len(particles))
        metadata["snapshot_times"].append(time)
        print(f"{label}: writing statistics/checkpoint", flush=True)
        if snapshot_maps:
            snapshot_maps["metadata"] = np.array(json.dumps({"config_hash": digest, "step": step, "time": time}))
            atomic_save(output.with_name(f"{output.stem}.step{step}.npz"), snapshot_maps)
        atomic_save(output, pack(moments, metadata))
        print(f"Checkpoint saved: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume an identical configuration without counting snapshots twice")
    parser.add_argument("--max-snapshots", type=int, help="Process at most this many additional snapshots")
    parser.add_argument("--progress-interval", type=float, default=5.0, help="Seconds between particle progress lines (default: 5; 0 prints every particle)")
    args = parser.parse_args()
    if args.max_snapshots is not None and args.max_snapshots < 1:
        parser.error("--max-snapshots must be positive")
    if not np.isfinite(args.progress_interval) or args.progress_interval < 0:
        parser.error("--progress-interval must be finite and nonnegative")
    run(args.config, args.resume, args.max_snapshots, args.progress_interval)


if __name__ == "__main__":
    main()
