"""Optional fused CPU kernels; no relaxed floating-point arithmetic."""

import os
from contextlib import contextmanager

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads


@contextmanager
def thread_budget(workers):
    """Share the FFT worker budget and restore the caller's Numba thread mask."""
    previous = get_num_threads()
    threads = workers if workers > 0 else max(1, (os.cpu_count() or 1) + workers + 1)
    set_num_threads(min(threads, config.NUMBA_NUM_THREADS))
    try:
        yield
    finally:
        set_num_threads(previous)


@njit(cache=True, parallel=True, fastmath=False)
def _fluid_difference(field, fluid, axis, dx, result, bad):
    nx, ny, nz = field.shape
    # Each plane owns its output and error flag; the inner loop is contiguous
    # for the C-layout state arrays assembled by Reader.
    for plane in prange(nx):
        x = np.int64(plane)  # Periodic subtraction needs signed indices.
        for y in range(ny):
            for z in range(nz):
                if not fluid[x, y, z]:
                    result[x, y, z] = 0.0
                    continue
                coordinate = x if axis == 0 else y if axis == 1 else z
                size = nx if axis == 0 else ny if axis == 1 else nz
                lo = coordinate - 1 if coordinate else size - 1
                hi = coordinate + 1 if coordinate + 1 < size else 0
                xm, ym, zm = (lo, y, z) if axis == 0 else (x, lo, z) if axis == 1 else (x, y, lo)
                xp, yp, zp = (hi, y, z) if axis == 0 else (x, hi, z) if axis == 1 else (x, y, hi)
                if fluid[xm, ym, zm] and fluid[xp, yp, zp]:
                    value = field[xp, yp, zp] - field[xm, ym, zm]
                else:
                    lo2 = coordinate - 2 if coordinate >= 2 else coordinate - 2 + size
                    hi2 = coordinate + 2 if coordinate + 2 < size else coordinate + 2 - size
                    xmm, ymm, zmm = (lo2, y, z) if axis == 0 else (x, lo2, z) if axis == 1 else (x, y, lo2)
                    xpp, ypp, zpp = (hi2, y, z) if axis == 0 else (x, hi2, z) if axis == 1 else (x, y, hi2)
                    if fluid[xp, yp, zp] and fluid[xpp, ypp, zpp]:
                        value = -3 * field[x, y, z] + 4 * field[xp, yp, zp] - field[xpp, ypp, zpp]
                    elif fluid[xm, ym, zm] and fluid[xmm, ymm, zmm]:
                        value = 3 * field[x, y, z] - 4 * field[xm, ym, zm] + field[xmm, ymm, zmm]
                    else:
                        bad[x] = True
                        value = 0.0
                result[x, y, z] = value / (2 * dx)


def fluid_difference(field, fluid, axis, dx, *, workers):
    result = np.empty_like(field)
    bad = np.zeros(field.shape[0], dtype=np.bool_)
    with thread_budget(workers):
        _fluid_difference(field, fluid, axis, dx, result, bad)
    if np.any(bad):
        raise ValueError("Fluid gap has no three-point derivative stencil; refine the grid or supply a different reconstruction")
    return result


@njit(cache=True, parallel=True, fastmath=False)
def _merge(count, weight, mean, m2, m3, m4, bcount, weight_b, mean_b, m2_b, m3_b, m4_b):
    for i in prange(mean.size):
        good = np.isfinite(mean_b[i]) and weight_b[i] > 0
        a = weight[i]
        b = weight_b[i] if good else 0.0
        total = a + b
        ra = a / total if total > 0 else 0.0
        rb = b / total if total > 0 else 0.0
        bmean = mean_b[i] if good else mean[i]
        delta = bmean - mean[i] if good and a > 0 else 0.0
        b2 = m2_b[i] if good else 0.0
        b3 = m3_b[i] if good else 0.0
        b4 = m4_b[i] if good else 0.0
        cross_weight = a * rb
        # Higher orders must read the OLD lower-order sums.
        m4[i] += b4 + delta**4 * cross_weight * (ra**2 - ra * rb + rb**2) + 6 * delta**2 * (ra**2 * b2 + rb**2 * m2[i]) + 4 * delta * (ra * b3 - rb * m3[i])
        m3[i] += b3 + delta**3 * cross_weight * (ra - rb) + 3 * delta * (ra * b2 - rb * m2[i])
        m2[i] += b2 + delta**2 * cross_weight
        mean[i] = mean[i] + delta * rb if a > 0 else bmean
        weight[i] = total
        count[i] += bcount[i] if good else 0


def merge_moments(result, other, *, workers):
    """Validate before mutation, then merge each map location in one pass."""
    if np.any(~np.isfinite(other.weight)) or np.any(other.weight < 0):
        raise ValueError("Statistical weights must be finite and nonnegative")
    names = ("count", "weight", "mean", "m2", "m3", "m4")
    arrays = [getattr(value, name).reshape(-1) for value in (result, other) for name in names]
    with thread_budget(workers):
        _merge(*arrays)
