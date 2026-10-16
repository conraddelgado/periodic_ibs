"""Periodic filtering, fluid-side differences, interpolation, and deposition."""

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


def fluid_derivative(field, fluid, axis, dx):
    """Second-order centered or one-sided difference without crossing solid cells."""
    minus = np.roll(field, 1, axis)
    plus = np.roll(field, -1, axis)
    fm = np.roll(fluid, 1, axis)
    fp = np.roll(fluid, -1, axis)
    centered = fluid & fm & fp
    forward = fluid & ~centered & fp & np.roll(fluid, -2, axis)
    backward = fluid & ~centered & ~forward & fm & np.roll(fluid, 2, axis)
    if np.any(fluid & ~(centered | forward | backward)):
        raise ValueError("Fluid gap has no three-point derivative stencil; refine the grid or supply a different reconstruction")
    result = np.zeros_like(field)
    result[centered] = ((plus - minus) / (2 * dx))[centered]
    result[forward] = ((-3 * field + 4 * plus - np.roll(field, -2, axis)) / (2 * dx))[forward]
    result[backward] = ((3 * field - 4 * minus + np.roll(field, 2, axis)) / (2 * dx))[backward]
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
