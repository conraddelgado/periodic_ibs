"""Particle-centered maps and mergeable population moments."""

import numpy as np


class Moments:
    """Weighted, pointwise parallel-Welford updates, ignoring undefined samples."""

    def __init__(self, shape):
        self.weight = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)

    def update(self, value, weight=1.0, within_m2=0.0):
        weight = np.broadcast_to(weight, self.mean.shape)
        good = np.isfinite(value) & (weight > 0)
        new_weight = np.where(good, weight, 0)
        total = self.weight + new_weight
        delta = np.where(good, value, self.mean) - self.mean
        fraction = np.divide(new_weight, total, out=np.zeros_like(total), where=total > 0)
        self.m2 += np.where(good, within_m2, 0) + delta**2 * self.weight * fraction
        self.mean += delta * fraction
        self.weight = total

    @property
    def variance(self):
        return np.divide(self.m2, self.weight, out=np.full_like(self.m2, np.nan), where=self.weight > 0)


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
