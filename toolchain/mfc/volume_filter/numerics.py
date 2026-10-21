"""Periodic filtering, fluid-side differences, interpolation, and deposition."""

from collections import OrderedDict
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import fft


@dataclass
class Grid:
    """Uniform cell-centered Cartesian grid; origin is the lower domain face."""

    shape: tuple
    lengths: tuple
    origin: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if len(self.shape) != 3 or any(int(n) != n or n < 5 for n in self.shape):
            raise ValueError("grid.shape must contain three cell counts >= 5")
        self.shape = tuple(int(n) for n in self.shape)
        if len(self.lengths) != 3 or not np.all(np.isfinite(self.lengths)) or min(self.lengths) <= 0:
            raise ValueError("grid.lengths must contain three positive lengths")
        if len(self.origin) != 3 or not np.all(np.isfinite(self.origin)):
            raise ValueError("grid.origin must contain three finite coordinates")
        self.dx = np.asarray(self.lengths) / self.shape
        self.dv = float(np.prod(self.dx))

    def delta(self, x, center):
        """Minimum-image displacement, including particles crossing a box face."""
        lengths = np.asarray(self.lengths)
        return (x - center + lengths / 2) % lengths - lengths / 2

    def stencil(self, points):
        q = ((np.asarray(points) - self.origin) / self.dx - 0.5) % self.shape
        lo = np.floor(q).astype(np.int64)
        fraction = q - lo
        for corner in product((0, 1), repeat=3):
            index = (lo + corner) % self.shape
            weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=-1)
            yield tuple(index.T), weight

    def sample(self, field, points, valid=None):
        """Periodic trilinear interpolation; reject any nonzero invalid contribution."""
        result = np.zeros(len(points))
        good = np.ones(len(points), dtype=bool)
        for index, weight in self.stencil(points):
            active = weight > 1e-14
            values = field[index]
            good &= ~active | np.isfinite(values)
            if valid is not None:
                good &= ~active | valid[index]
            result += weight * np.where(np.isfinite(values), values, 0)
        return np.where(good, result, np.nan)

    def sample_cartesian(self, field, axes, valid=None):
        """Sample a tensor-product grid without materializing pointwise 3D stencils.

        Keep interpolation indices/fractions in 1D, then broadcast the eight
        corner contributions. Invalid support uses the same rule as sample().
        """
        stencils = []
        for dimension, coordinates in enumerate(axes):
            q = ((coordinates - self.origin[dimension]) / self.dx[dimension] - 0.5) % self.shape[dimension]
            lo = np.floor(q).astype(np.int64)
            fraction = q - lo
            stencils.append(((lo, (lo + 1) % self.shape[dimension]), (1 - fraction, fraction)))
        shape = tuple(len(coordinates) for coordinates in axes)
        result = np.zeros(shape)
        good = np.ones(shape, dtype=bool)
        for corner in product((0, 1), repeat=3):
            index = np.ix_(*(stencils[d][0][c] for d, c in enumerate(corner)))
            wx, wy, wz = (stencils[d][1][c] for d, c in enumerate(corner))
            weight = (wx[:, None, None] * wy[None, :, None]) * wz[None, None, :]
            values = field[index]
            finite = np.isfinite(values)
            active = weight > 1e-14
            good &= ~active | finite
            if valid is not None:
                good &= ~active | valid[index]
            result += weight * np.where(finite, values, 0)
        return np.where(good, result, np.nan)

    def deposit(self, points, strengths):
        """Conservative cloud-in-cell surface-source deposition (per unit volume)."""
        result = np.zeros(self.shape)
        for index, weight in self.stencil(points):
            np.add.at(result, index, weight * strengths / self.dv)
        return result


def sphere_mask(grid, particles):
    """Cell-center indicator; visit only each sphere's periodic bounding box."""
    fluid = np.ones(grid.shape, dtype=bool)
    for particle in particles:
        center, radius = particle[16:19], particle[19]
        if radius <= 0 or 2 * radius >= min(grid.lengths):
            raise ValueError("Sphere radius must be positive and smaller than half every box length")
        axes, distance = [], []
        for axis in range(3):
            coordinate = grid.origin[axis] + (np.arange(grid.shape[axis]) + 0.5) * grid.dx[axis]
            d = (coordinate - center[axis] + grid.lengths[axis] / 2) % grid.lengths[axis] - grid.lengths[axis] / 2
            index = np.flatnonzero(np.abs(d) <= radius)
            axes.append(index)
            distance.append(d[index])
        box = np.ix_(*axes)
        r2 = distance[0][:, None, None] ** 2 + distance[1][None, :, None] ** 2 + distance[2][None, None, :] ** 2
        fluid[box] &= r2 > radius**2
    return fluid


class LocalFluidFit:
    """Sparse affine fits using only nearby fluid cells, in cell-index units.

    Centering the fit at its weighted centroid preserves constants even when
    the samples cannot determine all three gradient components. The minimum
    norm fit sets undetermined gradient components to zero; it does not resolve
    subcell gaps. Geometry and weights are reusable across fields.
    """

    def __init__(self, fluid, coordinates, derivative=None):
        if not np.any(fluid):
            raise ValueError("Local reconstruction requires at least one fluid cell")
        coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3) % fluid.shape
        self.derivative = derivative
        self.indices, self.weights = [], []
        self.rank_deficient = self.expanded = 0
        shape = np.asarray(fluid.shape)
        pending = np.arange(len(coordinates))
        radius = 2
        while len(pending):
            offsets = np.array(list(product(*(range(-min(radius, n // 2), min(radius, (n - 1) // 2) + 1) for n in shape))))
            unresolved = []
            chunk = max(1, 32768 // len(offsets))
            for first in range(0, len(pending), chunk):
                rows = pending[first : first + chunk]
                points = coordinates[rows]
                index = (np.floor(points + 0.5).astype(np.int64)[:, None, :] + offsets) % shape
                delta = (index - points[:, None, :] + shape / 2) % shape - shape / 2
                good = fluid[tuple(np.moveaxis(index, -1, 0))]
                supported = np.any(good, axis=1)
                unresolved.extend(rows[~supported])
                if not np.any(supported):
                    continue
                rows, index, delta, good = rows[supported], index[supported], delta[supported], good[supported]
                weight = good / (1 + np.sum(delta**2, axis=-1))
                centroid = np.sum(weight[..., None] * delta, axis=1) / weight.sum(axis=1)[:, None]
                design = np.concatenate((np.ones((*weight.shape, 1)), delta - centroid[:, None, :]), axis=-1)
                root = np.sqrt(weight)
                left, singular, right = np.linalg.svd(design * root[..., None], full_matrices=False)
                retained = singular > 1e-12 * singular[:, :1]
                inverse = np.divide(1, singular, out=np.zeros_like(singular), where=retained)
                coefficients = ((right.swapaxes(1, 2) * inverse[:, None, :]) @ left.swapaxes(1, 2)) * root[:, None, :]
                if derivative is None:
                    weights = coefficients[:, 0] - np.einsum("ni,nij->nj", centroid, coefficients[:, 1:])
                else:
                    weights = coefficients[:, derivative + 1]
                self.rank_deficient += int(np.count_nonzero(retained.sum(axis=1) < 4))
                self.expanded += len(rows) if radius > 2 else 0
                flat = np.ravel_multi_index(tuple(np.moveaxis(index, -1, 0)), shape)
                anchor = flat[np.arange(len(rows)), good.argmax(axis=1)]
                self.indices.append((rows, flat, anchor))
                self.weights.append(weights)
            pending = np.asarray(unresolved, dtype=np.int64)
            radius *= 2
        self.size = len(coordinates)

    def __call__(self, field):
        result = np.empty(self.size)
        flat = np.asarray(field).ravel()
        for (rows, index, anchor), weights in zip(self.indices, self.weights):
            values, reference = flat[index], flat[anchor]
            active = weights != 0
            if np.any(active & ~np.isfinite(values)) or not np.all(np.isfinite(reference)):
                raise ValueError("Nonfinite fluid values in local reconstruction")
            difference = np.where(active, values, reference[:, None]) - reference[:, None]
            result[rows] = np.sum(weights * difference, axis=1)
            if self.derivative is None:
                result[rows] += reference
        return result


def fluid_derivative(field, fluid, axis, dx, fallbacks=None):
    """Prefer three-point differences; fit underresolved gaps without solid data.

    Two-cell runs use first-order differences. An isolated cell along this
    axis uses a local fluid-only affine fit. Optional per-state fallbacks cache
    that geometry and retain support counts, once per axis rather than field.
    """
    minus = np.roll(field, 1, axis)
    plus = np.roll(field, -1, axis)
    fm = np.roll(fluid, 1, axis)
    fp = np.roll(fluid, -1, axis)
    centered = fluid & fm & fp
    forward = fluid & ~centered & fp & np.roll(fluid, -2, axis)
    backward = fluid & ~centered & ~forward & fm & np.roll(fluid, 2, axis)
    missing = fluid & ~(centered | forward | backward)
    first_forward = missing & fp
    first_backward = missing & ~first_forward & fm
    isolated = missing & ~(first_forward | first_backward)
    result = np.zeros_like(field)
    result[centered] = (plus[centered] - minus[centered]) / (2 * dx)
    result[forward] = (-3 * field[forward] + 4 * plus[forward] - np.roll(field, -2, axis)[forward]) / (2 * dx)
    result[backward] = (3 * field[backward] - 4 * minus[backward] + np.roll(field, 2, axis)[backward]) / (2 * dx)
    result[first_forward] = (plus[first_forward] - field[first_forward]) / dx
    result[first_backward] = (field[first_backward] - minus[first_backward]) / dx
    cached = fallbacks.get(axis) if fallbacks is not None else None
    if cached is None:
        fit = LocalFluidFit(fluid, np.argwhere(isolated), derivative=axis) if np.any(isolated) else None
        cached = {
            "first_order_cells": int(np.count_nonzero(first_forward | first_backward)),
            "affine_fit_cells": int(np.count_nonzero(isolated)),
            "rank_deficient_fits": fit.rank_deficient if fit is not None else 0,
            "fit": fit,
        }
        if fallbacks is not None:
            fallbacks[axis] = cached
    if cached["fit"] is not None:
        result[isolated] = cached["fit"](field) / dx
    return result


class Gaussian:
    """Periodized continuum Gaussian multiplier; sigma is a dimensional std. dev."""

    def __init__(self, grid, sigma, workers=1):
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Filter sigma must be positive")
        self.grid, self.workers = grid, workers
        self.wave = []
        k2 = np.zeros((grid.shape[0], grid.shape[1], grid.shape[2] // 2 + 1))
        for axis in range(3):
            frequency = fft.rfftfreq if axis == 2 else fft.fftfreq
            wave = 2 * np.pi * frequency(grid.shape[axis], grid.dx[axis])
            shape = [1, 1, 1]
            shape[axis] = len(wave)
            k2 += wave.reshape(shape) ** 2
            # A real collocated field's odd derivative has zero Nyquist coefficient.
            if grid.shape[axis] % 2 == 0:
                wave[grid.shape[axis] // 2] = 0
            self.wave.append(wave.reshape(shape))
        self.kernel = np.exp(-0.5 * sigma**2 * k2)

    def __call__(self, field):
        spectrum = fft.rfftn(field, workers=self.workers)
        spectrum *= self.kernel
        return fft.irfftn(spectrum, s=self.grid.shape, workers=self.workers)

    def derivative(self, field, axis):
        spectrum = fft.rfftn(field, workers=self.workers)
        spectrum *= 1j * self.wave[axis]
        return fft.irfftn(spectrum, s=self.grid.shape, workers=self.workers)


class SpectralDerivatives:
    """Reuse at most two forward FFTs of immutable filtered closure fields."""

    def __init__(self, gaussian):
        self.gaussian = gaussian
        self.spectra = OrderedDict()

    def derivative(self, field, axis):
        key = id(field)
        if key not in self.spectra:
            if len(self.spectra) == 2:
                self.spectra.popitem(last=False)
            # Retain the field as well: its identity cannot be recycled while cached.
            self.spectra[key] = (field, fft.rfftn(field, workers=self.gaussian.workers))
        self.spectra.move_to_end(key)
        spectrum = self.spectra[key][1]
        return fft.irfftn(spectrum * (1j * self.gaussian.wave[axis]), s=self.gaussian.grid.shape, workers=self.gaussian.workers)

    def clear(self):
        self.spectra.clear()
