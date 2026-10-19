"""Accelerated maps and cached FFT derivatives against numerical references."""

import builtins
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import fft

from .closures import closure_fields, reconstruct
from .maps import Moments, ParticleMap
from .numerics import Gaussian, Grid, SpectralDerivatives, _fluid_derivative_numpy, fluid_derivative, sphere_mask
from .test_moments import accumulate
from .test_volume_filter import PHYSICS, SURFACE, conservative, particles

HAS_NUMBA = importlib.util.find_spec("numba") is not None


class AccelerationTests(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMBA, "optional numba dependency is not installed")
    def test_fluid_stencil_matches_numpy_all_axes_layouts_and_solid_support(self):
        from numba import get_num_threads

        grid = Grid((17, 19, 21), (2, 3, 4), (-1, -1.5, -2))
        states = particles(center=(-0.99, -1.49, -1.99), radius=0.35)
        fluid = sphere_mask(grid, states)
        field = np.random.default_rng(82).normal(size=grid.shape)
        field[~fluid] = np.nan
        previous = get_num_threads()
        for values in (field, np.asfortranarray(field), field[::-1]):
            for axis in range(3):
                reference = _fluid_derivative_numpy(values, fluid, axis, grid.dx[axis])
                result = fluid_derivative(values, fluid, axis, grid.dx[axis], workers=2)
                assert_allclose(result, reference, rtol=1e-14, atol=1e-14, equal_nan=True)
                self.assertEqual(get_num_threads(), previous)
        for axis in range(3):
            bad = np.zeros(grid.shape, dtype=bool)
            bad[0, 0, 0] = True
            with self.assertRaisesRegex(ValueError, "three-point"):
                fluid_derivative(field, bad, axis, grid.dx[axis], workers=2)
            self.assertEqual(get_num_threads(), previous)

    @unittest.skipUnless(HAS_NUMBA, "optional numba dependency is not installed")
    def test_fused_merges_preserve_weighted_scatter_empty_groups_and_inputs(self):
        from numba import get_num_threads

        rng = np.random.default_rng(92)
        samples = rng.exponential(size=(19, 80, 65))
        samples[:, 0, 0] = np.nan
        samples[::2, 1, 1] = np.nan
        samples[:, 2, 2] = 1e12
        samples[:, 3, 3] += 1e9
        weights = rng.uniform(0, 2, size=samples.shape)
        weights[::3] = 0
        groups = [accumulate(sample, weight) for sample, weight in zip(np.split(samples, [1, 5, 13]), np.split(weights, [1, 5, 13]))]
        names = ("count", "weight", "mean", "m2", "m3", "m4", "variance", "skewness", "kurtosis")
        previous = get_num_threads()
        for order in ((0, 1, 2, 3), (3, 1, 0, 2)):
            result, reference = Moments((80, 65)), Moments((80, 65))
            for index in order:
                group = groups[index]
                originals = {name: getattr(group, name).copy() for name in names[:6]}
                result.merge(group, workers=2)
                reference._combine(group.mean, group.weight, group.count, group.m2, group.m3, group.m4)
                for name in names:
                    assert_allclose(getattr(result, name), getattr(reference, name), rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)
                for name, original in originals.items():
                    assert_array_equal(getattr(group, name), original)
                self.assertEqual(get_num_threads(), previous)
            result.merge(Moments(result.mean.shape), workers=2)
            for name in names:
                assert_allclose(getattr(result, name), getattr(reference, name), rtol=1e-12, atol=1e-12, equal_nan=True)
        for weight in (-1, np.inf, np.nan):
            invalid = Moments((80, 65))
            invalid.weight[0, 0] = weight
            with self.assertRaisesRegex(ValueError, "weights"):
                result.merge(invalid, workers=2)
            assert_array_equal(result.count, reference.count)
            self.assertEqual(get_num_threads(), previous)

    @unittest.skipUnless(HAS_NUMBA, "optional numba dependency is not installed")
    def test_fused_maps_match_numpy_with_periodic_and_invalid_support(self):
        from numba import get_num_threads

        grid = Grid((17, 19, 21), (2, 3, 4), (-1, -1.5, -2))
        rng = np.random.default_rng(47)
        states = np.concatenate([particles(center=center) for center in rng.uniform(-5, 5, (13, 3))])
        states[0, 16:19] = np.asarray(grid.origin) + 3.5 * grid.dx
        field = rng.normal(size=grid.shape)
        field[3, 3, 4] = np.nan  # Zero-weight neighbor at the central map point.
        field[10, 11, 12] = np.inf
        field[1:3, 2:4, 3:5] = np.nan
        options = {"kind": "box", "length": 0.4, "points": 5}
        previous = get_num_threads()
        for data in (field, np.asfortranarray(field), np.full(grid.shape, np.nan), np.full(grid.shape, 1e12)):
            reference = ParticleMap(grid, dict(options, backend="numpy")).particles(data, states)
            result = ParticleMap(grid, dict(options, backend="numba"), workers=2).particles(data, states)
            self.assertEqual(get_num_threads(), previous)
            for name in ("count", "weight"):
                assert_array_equal(getattr(result, name), getattr(reference, name))
            for name in ("mean", "m2", "m3", "m4", "variance", "skewness", "kurtosis"):
                assert_allclose(getattr(result, name), getattr(reference, name), rtol=1e-10, atol=1e-11, equal_nan=True, err_msg=name)
        one = ParticleMap(grid, dict(options, backend="numba")).particles(field, states[:1])
        self.assertEqual(one.count[2, 2, 2], 1)
        empty = ParticleMap(grid, dict(options, backend="numba")).particles(field, states[:0])
        assert_array_equal(empty.count, 0)

    def test_auto_fallback_and_explicit_missing_dependency(self):
        real_import = builtins.__import__

        def without_numba(name, *args, **kwargs):
            if name.endswith(("numba_mapping", "numba_kernels")):
                raise ModuleNotFoundError("No module named numba", name="numba")
            return real_import(name, *args, **kwargs)

        grid = Grid((8, 8, 8), (1, 1, 1))
        options = {"kind": "box", "length": 0.2, "points": 3}
        with patch("builtins.__import__", side_effect=without_numba):
            mapper = ParticleMap(grid, options)
            self.assertEqual(mapper.backend, "numpy")
            assert_allclose(mapper.particles(np.ones(grid.shape), particles()).mean, 1)
            with self.assertRaisesRegex(ValueError, "requires numba"):
                ParticleMap(grid, dict(options, backend="numba"))
            field = np.arange(8**3, dtype=float).reshape(grid.shape)
            fluid = np.ones(grid.shape, dtype=bool)
            assert_array_equal(fluid_derivative(field, fluid, 0, grid.dx[0]), _fluid_derivative_numpy(field, fluid, 0, grid.dx[0]))
            result, other = Moments((20, 20, 20)), Moments((20, 20, 20))
            result.update(np.ones(result.mean.shape))
            other.update(np.full(other.mean.shape, 3.0))
            result.merge(other)
            assert_allclose(result.mean, 2)
            assert_allclose(result.variance, 1)

    def test_spectral_cache_preserves_derivatives_and_is_bounded(self):
        grid = Grid((16, 18, 20), (2, 3, 4))
        gaussian = Gaussian(grid, 0.1)
        cache = SpectralDerivatives(gaussian)
        fields = np.random.default_rng(7).normal(size=(4, *grid.shape))
        for field in fields:
            original = field.copy()
            for axis in (0, 1, 2, 0):
                assert_allclose(cache.derivative(field, axis), gaussian.derivative(field, axis), rtol=1e-13, atol=1e-13)
                self.assertLessEqual(len(cache.spectra), 2)
            assert_array_equal(field, original)
        cache.clear()
        self.assertFalse(cache.spectra)

    def test_intrinsic_companions_preserve_mask_and_alpha_weighting(self):
        grid = Grid((24, 24, 24), (1, 1, 1))
        x = np.broadcast_to((np.arange(24) + 0.5)[:, None, None] / 24, grid.shape)
        velocity = [0.2 + 0.05 * np.sin(2 * np.pi * x), np.zeros(grid.shape), np.zeros(grid.shape)]
        state = reconstruct(grid, iter(conservative(np.ones(grid.shape), velocity, 2 + 0.1 * x)), particles(), PHYSICS)
        fields = dict(closure_fields(grid, state, particles(), 0.04, {"surface": SURFACE, "alpha_min": 0.9}))
        valid = fields["alpha"] >= 0.9
        self.assertTrue(np.any(~valid))
        self.assertTrue(np.any(valid))
        for name, field in fields.items():
            if name.startswith("alpha_R_"):
                intrinsic = fields[name.removeprefix("alpha_")]
                assert_array_equal(np.isnan(field), ~valid)
                assert_array_equal(np.isnan(intrinsic), ~valid)
                assert_allclose(field[valid], fields["alpha"][valid] * intrinsic[valid], rtol=1e-14, atol=1e-14)

    def test_all_closures_match_uncached_fft_reference(self):
        class Uncached:
            def __init__(self, gaussian):
                self.gaussian = gaussian

            def derivative(self, field, axis):
                return self.gaussian.derivative(field, axis)

            def clear(self):
                pass

        grid = Grid((24, 24, 24), (1, 1, 1))
        x = (np.arange(24) + 0.5)[:, None, None] / 24
        y, z = x.transpose(1, 0, 2), x.transpose(1, 2, 0)
        rho = np.broadcast_to(1 + 0.05 * np.sin(2 * np.pi * x) + 0.02 * np.cos(2 * np.pi * z), grid.shape)
        velocity = [
            np.broadcast_to(0.2 + 0.03 * np.cos(2 * np.pi * x) + 0.01 * np.sin(2 * np.pi * y), grid.shape),
            np.broadcast_to(0.04 * np.sin(2 * np.pi * y) + 0.02 * np.cos(2 * np.pi * z), grid.shape),
            np.broadcast_to(0.03 * np.sin(2 * np.pi * z) + 0.01 * np.cos(2 * np.pi * x), grid.shape),
        ]
        pressure = np.broadcast_to(2 + 0.1 * np.sin(2 * np.pi * x) + 0.05 * np.sin(2 * np.pi * y), grid.shape)
        physics = dict(PHYSICS, mu=0.03, conductivity=0.04)
        state = reconstruct(grid, iter(conservative(rho, velocity, pressure)), particles(), physics)
        for sigma in (0.08, 0.12):
            with patch("mfc.volume_filter.numerics.fft.rfftn", wraps=fft.rfftn) as forward:
                result = dict(closure_fields(grid, state, particles(), sigma, {"surface": dict(SURFACE, thermal_bc="reconstruct")}))
                cached_calls = forward.call_count
            with (
                patch("mfc.volume_filter.closures.SpectralDerivatives", Uncached),
                patch("mfc.volume_filter.closures.fluid_derivative", side_effect=lambda f, mask, axis, dx, **kw: _fluid_derivative_numpy(f, mask, axis, dx)),
                patch("mfc.volume_filter.numerics.fft.rfftn", wraps=fft.rfftn) as forward,
            ):
                reference = dict(closure_fields(grid, state, particles(), sigma, {"surface": dict(SURFACE, thermal_bc="reconstruct")}))
                self.assertLess(cached_calls, forward.call_count - 10)
            self.assertEqual(set(result), set(reference))
            self.assertEqual(len(result), 59)
            for name in result:
                assert_allclose(result[name], reference[name], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)


if __name__ == "__main__":
    unittest.main()
