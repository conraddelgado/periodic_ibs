"""Particle-centered maps and mergeable population moments."""

import numpy as np


class Moments:
    """Mergeable weighted central sums through order four and valid-sample counts.

    Pairwise central-moment identities avoid subtracting large raw moments.
    See Pebay, SAND2008-6212, https://doi.org/10.2172/1028931.
    """

    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=np.int64)
        self.weight = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)
        self.m3 = np.zeros(shape)
        self.m4 = np.zeros(shape)

    def update(self, value, weight=1.0):
        """Add one realization per map location; ignore nonfinite observations."""
        self._combine(value, weight, 1, 0.0, 0.0, 0.0)

    def merge(self, other):
        """Combine complete distributions, including their within-group scatter."""
        if self.mean.shape != other.mean.shape:
            raise ValueError("Cannot merge moments with different map shapes")
        if self is other:
            raise ValueError("Cannot merge a moment accumulator with itself")
        self._combine(other.mean, other.weight, other.count, other.m2, other.m3, other.m4)

    def _combine(self, value, weight, count, m2, m3, m4):
        weight = np.broadcast_to(weight, self.mean.shape)
        if np.any(~np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("Statistical weights must be finite and nonnegative")
        good = np.isfinite(value) & (weight > 0)
        b = np.where(good, weight, 0)
        total = self.weight + b
        ra = np.divide(self.weight, total, out=np.zeros_like(total), where=total > 0)
        rb = np.divide(b, total, out=np.zeros_like(total), where=total > 0)
        bmean = np.where(good, value, self.mean)
        # Empty groups have no displacement contribution; avoid large powers of
        # an arbitrary empty-group mean when the first realization is added.
        delta = np.where(good & (self.weight > 0), bmean - self.mean, 0)
        b2, b3, b4 = (np.where(good, moment, 0) for moment in (m2, m3, m4))
        cross_weight = self.weight * rb
        # Update high orders first: every formula needs the OLD lower-order sums.
        self.m4 += b4 + delta**4 * cross_weight * (ra**2 - ra * rb + rb**2) + 6 * delta**2 * (ra**2 * b2 + rb**2 * self.m2) + 4 * delta * (ra * b3 - rb * self.m3)
        self.m3 += b3 + delta**3 * cross_weight * (ra - rb) + 3 * delta * (ra * b2 - rb * self.m2)
        self.m2 += b2 + delta**2 * cross_weight
        self.mean = np.where(self.weight > 0, self.mean + delta * rb, bmean)
        self.weight = total
        self.count += np.where(good, count, 0).astype(np.int64)

    def central_moment(self, order):
        if order not in (2, 3, 4):
            raise ValueError("Central moment order must be 2, 3, or 4")
        return np.divide(getattr(self, f"m{order}"), self.weight, out=np.full_like(self.mean, np.nan), where=self.weight > 0)

    @property
    def variance(self):
        return self.central_moment(2)

    @property
    def skewness(self):
        variance = self.variance
        return np.divide(self.central_moment(3), variance**1.5, out=np.full_like(self.mean, np.nan), where=variance > 0)

    @property
    def kurtosis(self):
        """Pearson kurtosis (Gaussian value 3), undefined for zero variance."""
        variance = self.variance
        return np.divide(self.central_moment(4), variance**2, out=np.full_like(self.mean, np.nan), where=variance > 0)

    def raw_moment(self, order):
        """E[X**order], recovered from the stably accumulated central moments."""
        mean = np.where(self.weight > 0, self.mean, np.nan)
        if order == 1:
            return mean
        if order == 2:
            return mean**2 + self.variance
        if order == 3:
            return mean**3 + 3 * mean * self.variance + self.central_moment(3)
        if order == 4:
            return mean**4 + 6 * mean**2 * self.variance + 4 * mean * self.central_moment(3) + self.central_moment(4)
        raise ValueError("Raw moment order must be between 1 and 4")


class ParticleMap:
    """Fixed laboratory axes, physical-length offsets, equal particle weights.

    Axisymmetric scalar-component maps average each ring before particle/time
    statistics; vector/tensor components remain Cartesian (explicitly labelled).
    """

    def __init__(self, grid, config):
        self.grid, self.config = grid, config
        self.mode = config["kind"]
        if self.mode == "box":
            axes = [np.asarray(config[name], dtype=float) for name in ("x", "y", "z")]
            if any(a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) for a in axes):
                raise ValueError("Box map axes must be finite nonempty lists")
            self.shape = tuple(len(a) for a in axes)
            self.offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        elif self.mode == "axisymmetric":
            axial = np.asarray(config["axial"], dtype=float)
            radial = np.asarray(config["radial"], dtype=float)
            ntheta = config.get("azimuthal_points", 32)
            axis = config.get("axis", 0)
            if axis not in (0, 1, 2) or not isinstance(ntheta, int) or ntheta < 4:
                raise ValueError("Invalid axis or azimuthal_points")
            if axial.ndim != 1 or radial.ndim != 1 or not axial.size or not radial.size or not np.all(np.isfinite(axial)) or not np.all(np.isfinite(radial)) or np.any(radial < 0):
                raise ValueError("Invalid axial/radial map coordinates")
            x, r, theta = np.meshgrid(axial, radial, np.arange(ntheta) * (2 * np.pi / ntheta), indexing="ij")
            offsets = np.zeros((*x.shape, 3))
            transverse = [i for i in range(3) if i != axis]
            offsets[..., axis] = x
            offsets[..., transverse[0]] = r * np.cos(theta)
            offsets[..., transverse[1]] = r * np.sin(theta)
            self.shape = (len(axial), len(radial))
            self.offsets = offsets.reshape(-1, 3)
            self.ntheta = ntheta
        else:
            raise ValueError("map.kind must be box or axisymmetric")
        if np.any(np.max(np.abs(self.offsets), axis=0) >= np.asarray(grid.lengths) / 2):
            raise ValueError("Map offsets must stay within half the box length to avoid repeated periodic images")

    def particles(self, field, states):
        result = Moments(self.shape)
        for particle in states:
            value = self.grid.sample(field, particle[16:19] + self.offsets)
            if self.mode == "axisymmetric":
                # Reject an incomplete ring; do not bias an angular average by masking angles.
                value = np.mean(value.reshape(*self.shape, self.ntheta), axis=-1)
            else:
                value = value.reshape(self.shape)
            result.update(value)
        return result
