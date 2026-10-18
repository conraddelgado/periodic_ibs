"""Compare generic and Cartesian map interpolation plus moment accumulation.

Run from the repository root:
PYTHONPATH=toolchain python misc/volume_filter/benchmark_maps.py --points 100
This measures one scalar field, excluding DNS reads, closure FFTs, and output.
"""

import argparse
from time import perf_counter

import numpy as np

from mfc.volume_filter.maps import Moments, ParticleMap
from mfc.volume_filter.numerics import Grid


def measure(grid, mapper, field, centers, reference):
    result = Moments(mapper.shape)
    interpolation = updates = 0.0
    for center in centers:
        start = perf_counter()
        if reference:
            value = grid.sample(field, center + mapper.offsets).reshape(mapper.shape)
        else:
            value = grid.sample_cartesian(field, [mapper.box_axis + x for x in center])
        interpolation += perf_counter() - start
        start = perf_counter()
        if reference:
            result._combine(value, 1.0, 1, 0.0, 0.0, 0.0)  # Original general merge path.
        else:
            result.update(value)
        updates += perf_counter() - start
    return result, (interpolation, updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.particles < 1 or args.repeats < 1:
        parser.error("particles and repeats must be positive")
    rng = np.random.default_rng(81)
    grid = Grid((128, 128, 128), (1, 1, 1))
    mapper = ParticleMap(grid, {"kind": "box", "length": 0.4, "points": args.points})
    field = rng.normal(size=grid.shape)
    centers = rng.uniform(size=(args.particles, 3))
    timings = {True: [], False: []}
    for repeat in range(args.repeats):
        results = {}
        for reference in (True, False) if repeat % 2 == 0 else (False, True):
            results[reference], timing = measure(grid, mapper, field, centers, reference)
            timings[reference].append(timing)
        for name in ("count", "weight", "mean", "m2", "m3", "m4"):
            np.testing.assert_allclose(getattr(results[False], name), getattr(results[True], name), rtol=1e-11, atol=1e-12)
    print(f"Map: {mapper.shape}; particles: {args.particles}; scalar fields: 1; repeats: {args.repeats}")
    for reference in (True, False):
        times = np.asarray(timings[reference])
        label = "generic" if reference else "optimized"
        print(f"{label}: interpolation={np.median(times[:, 0]):.4f}s, moments={np.median(times[:, 1]):.4f}s, total={np.median(times.sum(axis=1)):.4f}s")
    print(f"Median mapping speedup: {np.median(np.sum(timings[True], axis=1)) / np.median(np.sum(timings[False], axis=1)):.2f}x")
    print("Counts and central moment sums agree within rtol=1e-11, atol=1e-12.")


if __name__ == "__main__":
    main()
