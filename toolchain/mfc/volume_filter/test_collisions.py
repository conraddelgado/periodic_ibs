"""Thin-gap reconstruction and pooled statistics for moving, colliding spheres."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from .__main__ import run
from .closures import Surface, closure_fields, reconstruct
from .maps import ParticleMap
from .numerics import Gaussian, Grid, LocalFluidFit, fluid_derivative, sphere_mask
from .test_volume_filter import SURFACE, config, conservative, particles


def pair(grid, gap=0.0, center=None, time=0.0):
    if center is None:
        center = (np.asarray(grid.shape) // 2 + 0.5) * grid.dx + grid.origin
    return np.concatenate([particles(center=np.asarray(center) + [sign * (0.15 + gap / 2), 0, 0], radius=0.15, time=time) for sign in (-1, 1)])


class CollisionTests(unittest.TestCase):
    def test_first_order_and_isolated_affine_derivatives_ignore_solid_values(self):
        grid = Grid((13, 15, 17), (1.3, 3, 5.1))
        coordinates = np.indices(grid.shape).transpose(1, 2, 3, 0) * grid.dx
        gradient = np.array([0.7, -1.2, 2.3])
        field = 4 + coordinates @ gradient
        center = np.array([6, 7, 8])
        for axis in range(3):
            with self.subTest(axis=axis):
                # A two-cell run has only a first-order fluid stencil.
                fluid = np.zeros(grid.shape, dtype=bool)
                other = center.copy()
                other[axis] += 1
                fluid[tuple(center)] = fluid[tuple(other)] = True
                dirty = np.where(fluid, field, np.nan)
                cache = {}
                actual = fluid_derivative(dirty, fluid, axis, grid.dx[axis], cache)
                assert_allclose(actual[fluid], gradient[axis], atol=2e-14)
                self.assertEqual(cache[axis]["first_order_cells"], 2)
                self.assertEqual(cache[axis]["affine_fit_cells"], 0)
                # No face neighbors in this direction, but off-axis fluid
                # samples determine the full local affine gradient.
                offsets = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
                offsets = np.roll(offsets, axis, axis=1)
                fluid[:] = False
                fluid[tuple((center + offsets).T)] = True
                cache = {}
                for poison in (np.nan, np.inf, 1e100):
                    dirty = np.where(fluid, field, poison)
                    actual = fluid_derivative(dirty, fluid, axis, grid.dx[axis], cache)
                    self.assertAlmostEqual(actual[tuple(center)], gradient[axis], places=12)
                    assert_allclose(actual[~fluid], 0, atol=0)

    def test_local_fits_preserve_constants_and_report_missing_information(self):
        fluid = np.zeros((9, 11, 13), dtype=bool)
        fluid[4, 5, 6] = True
        field = np.where(fluid, 7.0, np.nan)
        point = np.array([[4.2, 5.3, 6.1]])
        fit = LocalFluidFit(fluid, point)
        assert_allclose(fit(field), 7, atol=0)
        self.assertEqual(fit.rank_deficient, 1)
        translated = LocalFluidFit(fluid, point + [9, -11, 26])
        assert_allclose(translated(field), fit(field), atol=0)
        for axis in range(3):
            cache = {}
            assert_allclose(fluid_derivative(field, fluid, axis, 0.2, cache), 0, atol=0)
            self.assertEqual(cache[axis]["rank_deficient_fits"], 1)
        expanded = LocalFluidFit(fluid, [[0, 0, 0]])
        assert_allclose(expanded(field), 7, atol=0)
        self.assertEqual(expanded.expanded, 1)
        with self.assertRaisesRegex(ValueError, "fluid cell"):
            LocalFluidFit(np.zeros_like(fluid), point)
        with self.assertRaisesRegex(ValueError, "Nonfinite fluid"):
            fit(np.full(fluid.shape, np.nan))
        # Half-index ties must select identical neighborhoods after wrapping
        # through odd-sized dimensions, including for nonconstant fields.
        fluid[:] = True
        field = np.random.default_rng(19).normal(size=fluid.shape)
        points = np.array([[4.5, 5.5, 6.5], [0.5, 0, 0]])
        original = LocalFluidFit(fluid, points)
        shifted = LocalFluidFit(fluid, points + [9, -11, 26])
        assert_allclose(original(field), shifted(field), atol=1e-14)

    def test_contact_surface_traces_and_conservative_sources(self):
        grid = Grid((32, 32, 32), (1, 1, 1))
        x, y, z = [(np.indices(grid.shape)[axis] + 0.5) * grid.dx[axis] for axis in range(3)]
        for gap in (0.25 * grid.dx[0], 0.0, -0.5 * grid.dx[0]):
            with self.subTest(gap=gap):
                bodies = pair(grid, gap)
                bodies[:, 7:10] = [0.7, -0.2, 0.1]
                fluid = sphere_mask(grid, bodies)
                surface = Surface(grid, bodies, fluid, SURFACE)
                self.assertGreater(surface.metadata["affine_fit_points"], 0)
                pressure = np.where(fluid, 3 + 0.4 * x - 0.2 * y + 0.3 * z, np.nan)
                expected = 3 + surface.points @ [0.4, -0.2, 0.3]
                assert_allclose(surface.trace(pressure), expected, atol=1e-13)
                assert_allclose(surface.trace(np.where(fluid, 3, np.nan)), 3, atol=1e-14)
                surface.force = 3 * surface.normals
                sources = dict(surface.sources(Gaussian(grid, 0.08)))
                for name in ("F_x", "F_y", "F_z", "F_u"):
                    self.assertAlmostEqual(sources[name].sum() * grid.dv, 0, places=12)
                    self.assertTrue(np.all(np.isfinite(sources[name])))
                if gap < 0:
                    self.assertGreater(surface.metadata["buried_points"], 0)
                    self.assertLess(surface.areas.sum(), 8 * np.pi * 0.15**2)
                    for body in bodies:
                        radius2 = np.sum(grid.delta(surface.points, body[16:19]) ** 2, axis=1)
                        self.assertTrue(np.all(radius2 >= body[19] ** 2 - 1e-14))

    def test_periodic_surface_equivalence_and_reduced_offsets(self):
        grid = Grid((32, 32, 32), (1, 1, 1), (-0.5, -0.5, -0.5))
        settings = dict(SURFACE, polar_points=12)
        bodies = pair(grid, 0, center=[0.49, 0.015625, 0.015625])
        shifted = bodies.copy()
        shifted[:, 16:19] += [1, -2, 3]
        left = Surface(grid, bodies, sphere_mask(grid, bodies), settings)
        right = Surface(grid, shifted, sphere_mask(grid, shifted), settings)
        x = grid.origin[0] + (np.arange(32) + 0.5) * grid.dx[0]
        field = np.broadcast_to(3 + np.sin(2 * np.pi * x)[:, None, None], grid.shape)
        assert_allclose(left.trace(field), right.trace(field), atol=1e-12)
        self.assertEqual(left.metadata, right.metadata)
        # A gap can support a shorter pair even when the configured pair fails.
        reduced = Surface(grid, pair(grid, 0.08), sphere_mask(grid, pair(grid, 0.08)), settings)
        self.assertGreater(reduced.metadata["reduced_offset_points"], 0)
        assert_allclose(reduced.trace(np.ones(grid.shape)), 1, atol=1e-14)

    def test_contained_and_coincident_spheres_use_the_union_boundary(self):
        grid = Grid((24, 24, 24), (1, 1, 1))
        outer = particles(radius=0.15)
        reference = Surface(grid, outer, sphere_mask(grid, outer), SURFACE)
        reference.force = 3 * reference.normals
        expected = dict(reference.sources(Gaussian(grid, 0.08)))
        for inner in (outer.copy(), particles(radius=0.05)):
            bodies = np.concatenate((outer, inner))
            surface = Surface(grid, bodies, sphere_mask(grid, bodies), SURFACE)
            self.assertEqual(surface.metadata["buried_points"], len(reference.points))
            assert_allclose(surface.areas, reference.areas, atol=0)
            surface.force = 3 * surface.normals
            for name, value in surface.sources(Gaussian(grid, 0.08)):
                assert_allclose(value, expected[name], atol=1e-14)

    def test_moving_collision_statistics_match_direct_samples_and_spawned_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = copy.deepcopy(config())
            cfg.update(grid={"shape": [24, 24, 24], "lengths": [1, 1, 1]}, snapshots=[1, 2, 3], fft_workers=2, fields=["F_x", "R_mu_xx", "R_q_x", "rho_bar"], sigmas=[0.08, 0.12])
            cfg["physics"].update(mu=0.03, conductivity=0.07)
            cfg["surface"]["thermal_bc"] = "reconstruct"
            cfg["map"]["backend"] = "numpy"
            cfg["diagnostics"] = {"fields": cfg["fields"], "map_indices": [[0, 0, 0], [1, 1, 1]], "particle_loads": True}
            grid = Grid(**cfg["grid"])
            mapper = ParticleMap(grid, cfg["map"])
            x, y, z = [(np.indices(grid.shape)[axis] + 0.5) * grid.dx[axis] for axis in range(3)]
            samples = {(name, index): [] for name in cfg["fields"] for index in range(2)}
            for step, gap in enumerate((0.25 * grid.dx[0], 0.0, -0.25 * grid.dx[0]), 1):
                bodies = pair(grid, gap, center=[0.96 + 0.04 * step, 0.52, 0.52], time=step / 10)
                bodies[:, 7:10] = [0.2 * step, 0.1, 0.0]
                bodies[:, 10:13] = [0, 0, 0.1 * step]
                bodies[:, 1:7] = np.arange(12).reshape(2, 6) * step
                rho = 1 + 0.03 * step * np.sin(2 * np.pi * x)
                velocity = [0.2 * step + 0.02 * np.cos(2 * np.pi * x), 0.03 * np.sin(2 * np.pi * y), 0.01 * np.sin(2 * np.pi * z)]
                fields = conservative(rho, velocity, 3 + 0.2 * step + 0.1 * np.sin(2 * np.pi * y))
                # Invalid ghost data must not affect either reconstruction path.
                fluid = sphere_mask(grid, bodies)
                for field in fields:
                    field[~fluid] = np.nan
                np.concatenate([field.ravel(order="F") for field in fields]).tofile(root / f"restart_data/{step}.dat")
                bodies.tofile(root / f"restart_data/ib_state_{step}.dat")
                state = reconstruct(grid, iter([field.copy() for field in fields]), bodies, cfg["physics"])
                for index, sigma in enumerate(cfg["sigmas"]):
                    for name, field in closure_fields(grid, state, bodies, sigma, cfg):
                        self.assertTrue(np.all(np.isfinite(field)), name)
                        if name in cfg["fields"]:
                            for body in bodies:
                                samples[name, index].append(grid.sample(field, mapper.offsets + body[16:19]).reshape(mapper.shape))
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                run(path)
                cfg["output"] = "parallel.npz"
                cfg["map"]["backend"] = "auto"
                path.write_text(json.dumps(cfg))
                run(path, snapshot_workers=2)
            with np.load(root / "stats.npz") as serial, np.load(root / "parallel.npz") as parallel:
                for key in serial.files:
                    if key != "metadata":
                        assert_allclose(serial[key], parallel[key], rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=key)
                for (name, index), realizations in samples.items():
                    values = np.stack(realizations)
                    mean = values.mean(axis=0)
                    centered = values - mean
                    variance = np.mean(centered**2, axis=0)
                    expected = {"mean": mean, "variance": variance, "skewness": np.mean(centered**3, axis=0) / variance**1.5, "kurtosis": np.mean(centered**4, axis=0) / variance**2}
                    for statistic, value in expected.items():
                        assert_allclose(serial[f"{name}.{statistic}.sigma{index}"], value, rtol=1e-8, atol=1e-10)
                metadata = json.loads(str(serial["metadata"]))
                other = json.loads(str(parallel["metadata"]))
                self.assertEqual(metadata["reconstruction"], other["reconstruction"])
                self.assertEqual(metadata["completed"], 3)
                self.assertEqual(metadata["particle_statistics"]["realizations"], 6)
                self.assertTrue(any(record["surface"]["affine_fit_points"] for record in metadata["reconstruction"]["snapshots"]))
                assert_allclose(serial["diagnostics.F_x.sigma0.count"], 2)
                assert_allclose(serial["diagnostics.times"], [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
