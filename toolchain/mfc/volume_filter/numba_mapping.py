"""Fused box interpolation and particle moments; imported only when available."""

import os

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads


@njit(cache=True, parallel=True, fastmath=False)
def _box_moments(field, lo, hi, fraction, count, weight, means, second, third, fourth):
    points = lo.shape[2]
    for point in prange(means.size):
        ix = point // (points * points)
        iy = (point // points) % points
        iz = point % points
        n = 0
        mean = m2 = m3 = m4 = 0.0
        for particle in range(lo.shape[0]):
            x, y, z = lo[particle, 0, ix], lo[particle, 1, iy], lo[particle, 2, iz]
            x1, y1, z1 = hi[particle, 0, ix], hi[particle, 1, iy], hi[particle, 2, iz]
            fx, fy, fz = fraction[particle, 0, ix], fraction[particle, 1, iy], fraction[particle, 2, iz]
            value = 0.0
            good = True
            for cx in range(2):
                wx = fx if cx else 1 - fx
                for cy in range(2):
                    wy = fy if cy else 1 - fy
                    for cz in range(2):
                        wz = fz if cz else 1 - fz
                        w = (wx * wy) * wz
                        sample = field[x1 if cx else x, y1 if cy else y, z1 if cz else z]
                        if np.isfinite(sample):
                            value += w * sample
                        elif w > 1e-14:
                            good = False
            if not good or not np.isfinite(value):
                continue
            total = n + 1
            delta = value - mean if n else 0.0
            delta_n = delta / total
            delta_n2 = delta_n * delta_n
            term = delta * delta_n * n
            m4 += term * delta_n2 * (total * total - 3 * total + 3) + 6 * delta_n2 * m2 - 4 * delta_n * m3
            m3 += term * delta_n * (total - 2) - 3 * delta_n * m2
            m2 += term
            mean = mean + delta_n if n else value
            n = total
        count[point] = n
        weight[point] = n
        means[point] = mean
        second[point] = m2
        third[point] = m3
        fourth[point] = m4


def map_box(grid, axis, field, states, result, *, workers):
    """Small 1D stencils, no per-particle map arrays or shared reductions."""
    centers = np.asarray(states[:, 16:19], dtype=float)
    q = ((centers[:, :, None] + axis - np.asarray(grid.origin)[None, :, None]) / grid.dx[None, :, None] - 0.5) % np.asarray(grid.shape)[None, :, None]
    lo = np.floor(q).astype(np.int64)
    fraction = q - lo
    hi = (lo + 1) % np.asarray(grid.shape)[None, :, None]
    previous_threads = get_num_threads()
    threads = workers if workers > 0 else max(1, (os.cpu_count() or 1) + workers + 1)
    set_num_threads(min(threads, config.NUMBA_NUM_THREADS))
    try:
        arrays = tuple(getattr(result, name).ravel() for name in ("count", "weight", "mean", "m2", "m3", "m4"))
        _box_moments(np.asarray(field, dtype=float), lo, hi, fraction, *arrays)
    finally:
        set_num_threads(previous_threads)
