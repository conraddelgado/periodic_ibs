"""Accelerated maps and cached FFT derivatives against numerical references."""

import builtins
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import fft

from .closures import closure_fields, reconstruct
from .maps import ParticleMap
from .numerics import Gaussian, Grid, SpectralDerivatives
from .test_volume_filter import PHYSICS, SURFACE, conservative, particles

HAS_NUMBA = importlib.util.find_spec("numba") is not None


class AccelerationTests(unittest.TestCase):
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
            if name.endswith("numba_mapping"):
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
            with patch("mfc.volume_filter.closures.SpectralDerivatives", Uncached), patch("mfc.volume_filter.numerics.fft.rfftn", wraps=fft.rfftn) as forward:
                reference = dict(closure_fields(grid, state, particles(), sigma, {"surface": dict(SURFACE, thermal_bc="reconstruct")}))
                self.assertLess(cached_calls, forward.call_count - 10)
            self.assertEqual(set(result), set(reference))
            self.assertEqual(len(result), 59)
            for name in result:
                assert_allclose(result[name], reference[name], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)


if __name__ == "__main__":
    unittest.main()
