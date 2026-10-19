"""Compare warmed CPU fluid stencils and moment merges against NumPy references.

PYTHONPATH=toolchain python misc/volume_filter/benchmark_cpu.py --grid 400 --points 100 --workers 8
These isolated timings exclude FFTs, input/output, and particle mapping.
"""

import argparse
import copy
from time import perf_counter

import numpy as np

from mfc.volume_filter.maps import Moments
from mfc.volume_filter.numba_kernels import fluid_difference, merge_moments
from mfc.volume_filter.numerics import Grid, _fluid_derivative_numpy, sphere_mask


def report(label, timings):
    medians = {name: float(np.median(times)) for name, times in timings.items()}
    print(f"{label}: numpy {medians['numpy']:.4f}s, numba {medians['numba']:.4f}s; {medians['numpy'] / medians['numba']:.2f}x", flush=True)
    print(f"  runs: {timings}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.grid < 5 or args.points < 2 or min(args.workers, args.repeats) < 1:
        parser.error("grid >= 5, points >= 2, and positive workers/repeats are required")
    warm = np.ones((8, 8, 8))
    start = perf_counter()
    fluid_difference(warm, warm.astype(bool), 0, 0.1, workers=args.workers)
    merge_moments(Moments((8, 8, 8)), Moments((8, 8, 8)), workers=args.workers)
    print(f"Numba first-call setup/compilation/cache load: {perf_counter() - start:.3f}s", flush=True)
    rng = np.random.default_rng(92)
    grid = Grid((args.grid,) * 3, (1, 1, 1))
    states = np.zeros((1, 20))
    states[0, 16:20] = (0.02, 0.5, 0.5, 0.15)  # Periodic sphere; solid values are deliberately invalid.
    fluid = sphere_mask(grid, states)
    field = rng.normal(size=grid.shape)
    field[~fluid] = np.nan
    print(f"Source grid {grid.shape}; map {(args.points,) * 3}; threads {args.workers}; repeats {args.repeats}", flush=True)
    for axis in range(3):
        timings = {"numpy": [], "numba": []}
        for repeat in range(args.repeats):
            outputs = {}
            for backend in ("numpy", "numba") if repeat % 2 == 0 else ("numba", "numpy"):
                start = perf_counter()
                outputs[backend] = _fluid_derivative_numpy(field, fluid, axis, grid.dx[axis]) if backend == "numpy" else fluid_difference(field, fluid, axis, grid.dx[axis], workers=args.workers)
                timings[backend].append(perf_counter() - start)
            # Bound comparison scratch space as well as kernel scratch space.
            for first in range(0, args.grid, 8):
                np.testing.assert_allclose(outputs["numba"][first : first + 8], outputs["numpy"][first : first + 8], rtol=1e-14, atol=1e-14)
            del outputs
        report(f"Fluid derivative axis {axis}", timings)
    del field, fluid
    groups = []
    for realizations in (3, 5):
        value = Moments((args.points,) * 3)
        for _ in range(realizations):
            sample = rng.exponential(size=value.mean.shape)
            sample[0] = np.nan
            value.update(sample)
        groups.append(value)
    timings = {"numpy": [], "numba": []}
    for repeat in range(args.repeats):
        outputs = {}
        for backend in ("numpy", "numba") if repeat % 2 == 0 else ("numba", "numpy"):
            value = copy.deepcopy(groups[0])  # Reset outside the timed merge.
            other = groups[1]
            start = perf_counter()
            if backend == "numpy":
                value._combine(other.mean, other.weight, other.count, other.m2, other.m3, other.m4)
            else:
                merge_moments(value, other, workers=args.workers)
            timings[backend].append(perf_counter() - start)
            outputs[backend] = value
        for name in ("count", "weight", "mean", "m2", "m3", "m4", "variance", "skewness", "kurtosis"):
            np.testing.assert_allclose(getattr(outputs["numba"], name), getattr(outputs["numpy"], name), rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)
    report("Central-moment merge", timings)
    print("All derivative, count, and moment comparisons passed.")


if __name__ == "__main__":
    main()
