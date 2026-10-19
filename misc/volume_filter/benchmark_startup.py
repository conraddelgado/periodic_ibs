"""Time real snapshot reads and primitive reconstruction, without closure FFTs.

PYTHONPATH=toolchain python misc/volume_filter/benchmark_startup.py config.json
Add --legacy-mmap to compare the former shared-file read pattern. Run in
separate processes; filesystem cache state and contention affect comparisons.
"""

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

import numpy as np

from mfc.volume_filter.__main__ import validate
from mfc.volume_filter.closures import reconstruct
from mfc.volume_filter.numerics import Grid
from mfc.volume_filter.reader import Reader


def legacy_fields(reader, step, report):
    if reader.format != "shared":
        raise ValueError("--legacy-mmap comparison supports shared files only")
    count = int(np.prod(reader.grid.shape))
    for var in reader.indices:
        path = reader.path(reader.config["flow"], step)
        if path.stat().st_size != count * reader.config["nvars"] * reader.dtype.itemsize:
            raise ValueError(f"Incorrect restart size: {path}")
        report(f"legacy mmap reading conservative variable {var}")
        start = perf_counter()
        raw = np.memmap(path, dtype=reader.dtype, mode="r", offset=(var - 1) * count * reader.dtype.itemsize, shape=(count,))
        result = np.empty(reader.grid.shape)
        result[:] = raw.reshape(reader.grid.shape, order="F")
        del raw
        report(f"legacy variable {var} read/assembled in {perf_counter() - start:.2f}s")
        yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--step", type=int, help="Saved index (default: first configured snapshot)")
    parser.add_argument("--legacy-mmap", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve()
    config = json.loads(path.read_text())
    validate(config)
    grid = Grid(**config["grid"])
    reader = Reader(config["io"], grid, path.parent)
    step = config["snapshots"][0] if args.step is None else args.step
    start = perf_counter()

    def report(message):
        print(f"{perf_counter() - start:.2f}s: {message}", flush=True)

    report(f"step={step}, source grid={grid.shape}, reader={'legacy mmap' if args.legacy_mmap else 'sequential'}")
    particles = reader.particles(step)
    fields = legacy_fields(reader, step, report) if args.legacy_mmap else reader.fields(step, report)
    state = reconstruct(grid, fields, particles, config["physics"], report)
    report(f"Startup complete: {len(particles)} particles; pressure range={state.p.min():.6g}..{state.p.max():.6g}")
    print(f"Peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.3f} GiB (Linux)")
    print("No closure FFTs, particle maps, or output files were computed.")


if __name__ == "__main__":
    main()
