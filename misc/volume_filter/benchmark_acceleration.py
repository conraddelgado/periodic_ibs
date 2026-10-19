"""Compare warmed fused Numba maps with the existing Cartesian NumPy mapper.

PYTHONPATH=toolchain python misc/volume_filter/benchmark_acceleration.py --grid 400 --points 100 --particles 95 --workers 8
One scalar field only: excludes input reads, full-domain closures, and output.
"""

import argparse
from time import perf_counter

import numpy as np

from mfc.volume_filter.maps import ParticleMap
from mfc.volume_filter.numerics import Grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.particles, args.workers, args.repeats) < 1:
        parser.error("particles, workers, and repeats must be positive")
    # Warm the same array signatures with a small map before allocating test data.
    warm = ParticleMap(Grid((8, 8, 8), (1, 1, 1)), {"kind": "box", "length": 0.4, "points": 3, "backend": "numba"}, workers=args.workers)
    state = np.zeros((1, 20))
    start = perf_counter()
    warm.particles(np.ones((8, 8, 8)), state)
    print(f"Numba first-call setup/compilation/cache load: {perf_counter() - start:.3f}s", flush=True)
    rng = np.random.default_rng(81)
    grid = Grid((args.grid,) * 3, (1, 1, 1))
    field = rng.normal(size=grid.shape)
    states = np.zeros((args.particles, 20))
    states[:, 16:19] = rng.uniform(size=(args.particles, 3))
    mappers = {backend: ParticleMap(grid, {"kind": "box", "length": 0.4, "points": args.points, "backend": backend}, workers=args.workers) for backend in ("numpy", "numba")}
    timings = {backend: [] for backend in mappers}
    for repeat in range(args.repeats):
        results = {}
        for backend in ("numpy", "numba") if repeat % 2 == 0 else ("numba", "numpy"):
            start = perf_counter()
            results[backend] = mappers[backend].particles(field, states)
            timings[backend].append(perf_counter() - start)
        for name in ("count", "weight", "mean", "m2", "m3", "m4", "variance", "skewness", "kurtosis"):
            np.testing.assert_allclose(getattr(results["numba"], name), getattr(results["numpy"], name), rtol=1e-10, atol=1e-11, equal_nan=True, err_msg=name)
        del results
    print(f"Source grid: {grid.shape}; map: {(args.points,) * 3}; particles: {args.particles}; requested Numba threads: {args.workers}; repeats: {args.repeats}")
    medians = {backend: float(np.median(times)) for backend, times in timings.items()}
    for backend, median in medians.items():
        print(f"{backend}: median {median:.3f}s; runs {timings[backend]}")
    print(f"Mapping-only speedup: {medians['numpy'] / medians['numba']:.2f}x; counts and moments agree.")


if __name__ == "__main__":
    main()
