"""Hydrodynamic traction integrals and equally weighted particle/snapshot loads."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from .__main__ import HYDRO_PARTICLE_FIELDS, STATISTICS, process_snapshots, run, validate
from .closures import Surface, closure_fields, reconstruct
from .numerics import Gaussian, Grid, sphere_mask
from .test_volume_filter import PHYSICS, SURFACE, config, conservative, particles


class HydrodynamicParticleTests(unittest.TestCase):
    def test_owned_force_and_centroid_torque_across_periodic_boundaries(self):
        grid = Grid((24, 24, 24), (1, 1, 1))
        bodies = np.concatenate((particles(center=(0.98, 0.5, 0.5), radius=0.08), particles(center=(0.4, 0.5, 0.5), radius=0.12)))
        bodies[:, 7:10] = [[0.2, -0.3, 0.4], [-0.5, 0.6, -0.1]]
        bodies[:, 10:13] = [[0.7, -0.2, 0.5], [0.1, -0.4, 0.8]]
        a, b = np.array([0.7, -1.2, 0.4]), np.array([-0.3, 0.5, 0.8])
        expected = np.column_stack((4 * np.pi * bodies[:, 19, None] ** 2 * a, (8 * np.pi / 3) * bodies[:, 19, None] ** 4 * b))
        for shift in ([0, 0, 0], [1, -2, 3]):
            shifted = bodies.copy()
            shifted[:, 16:19] += shift
            surface = Surface(grid, shifted, sphere_mask(grid, shifted), SURFACE)
            # Prescribe fluid-on-solid traction a + b cross r. The source uses
            # its negative; the analytic torque integral is (8*pi/3)*R**4*b.
            surface.force = -(a + np.cross(b, surface.radials))
            loads = surface.particle_loads()
            assert_allclose(loads, expected, rtol=1e-13, atol=1e-15)
            sources = dict(surface.sources(Gaussian(grid, 0.1)))
            for column, axis in enumerate("xyz"):
                assert_allclose(sources[f"F_{axis}"].sum() * grid.dv, -loads[:, column].sum(), atol=1e-14)
            power = np.sum(shifted[:, 7:10] * loads[:, :3] + shifted[:, 10:13] * loads[:, 3:])
            assert_allclose(sources["F_u"].sum() * grid.dv, -power, atol=1e-14)

    def test_empty_and_buried_particles_preserve_record_order(self):
        grid = Grid((24, 24, 24), (1, 1, 1))
        outer = particles(radius=0.15)
        for inner in (outer.copy(), particles(radius=0.05)):
            bodies = np.concatenate((outer, inner))
            surface = Surface(grid, bodies, sphere_mask(grid, bodies), SURFACE)
            surface.force[:] = [-2, 3, -4]
            loads = surface.particle_loads()
            assert_allclose(loads[0, :3], 4 * np.pi * 0.15**2 * np.array([2, -3, 4]), atol=1e-14)
            assert_allclose(loads[1], 0, atol=0)
            assert_allclose(loads[:, 3:], 0, atol=1e-15)
        empty = np.empty((0, 20))
        surface = Surface(grid, empty, sphere_mask(grid, empty), SURFACE)
        self.assertEqual(surface.particle_loads().shape, (0, 6))

    def test_pressure_and_viscous_loads_match_analytic_integrals_and_fluid_sources(self):
        grid = Grid((32, 32, 32), (1, 1, 1))
        x, y, z = [(np.indices(grid.shape)[axis] + 0.5) * grid.dx[axis] for axis in range(3)]
        gradient = np.array([0.4, -0.2, 0.3])
        pressure = 10 + gradient[0] * x + gradient[1] * y + gradient[2] * z
        bodies = np.concatenate((particles(center=(0.3, 0.3, 0.3), radius=0.08), particles(center=(0.7, 0.7, 0.7), radius=0.1)))
        bodies[:, 1:7] = 1e9  # Saved net forces/torques cannot enter the answer.
        bodies[:, 7:10] = [0.3, -0.2, 0.1]
        bodies[:, 10:13] = [0.1, 0.2, -0.3]
        volumes = 4 * np.pi * bodies[:, 19] ** 3 / 3
        for mu in (0, 0.2):
            physics = dict(PHYSICS, mu=mu)
            # u=(xy,0,0) gives div(tau)=(0,mu/3,0); affine pressure gives
            # -grad(p)*volume. Both have zero centroid torque for a sphere.
            fields = conservative(np.ones(grid.shape), [x * y, np.zeros(grid.shape), np.zeros(grid.shape)], pressure)
            state = reconstruct(grid, iter(fields), bodies, physics)
            for sigma in (0.08, 0.14):
                mapped = dict(closure_fields(grid, state, bodies, sigma, {"surface": SURFACE}))
                expected = volumes[:, None] * (-gradient + [0, mu / 3, 0])
                assert_allclose(state.hydrodynamic_loads[:, :3], expected, atol=1e-13)
                assert_allclose(state.hydrodynamic_loads[:, 3:], 0, atol=1e-13)
                for column, axis in enumerate("xyz"):
                    assert_allclose(mapped[f"F_{axis}"].sum() * grid.dv, -expected[:, column].sum(), atol=1e-13)

    def test_pooled_moments_histories_and_workers_are_independent_of_filters_and_saved_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = copy.deepcopy(config())
            cfg.update(grid={"shape": [24, 24, 24], "lengths": [1, 1, 1]}, snapshots=[1, 2, 3], fields=["alpha"], sigmas=[0.08, 0.14], fft_workers=2)
            cfg["map"]["backend"] = "numpy"
            cfg["diagnostics"] = {"particle_loads": True}
            grid = Grid(**cfg["grid"])
            centers = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0.25, 0.75, 0.25)]
            coordinates = [(np.indices(grid.shape)[axis] + 0.5) * grid.dx[axis] for axis in range(3)]
            gradients = [[0.2, -0.1, 0.3], [0.8, -0.4, 0.1], [-0.3, 0.6, 1.0]]
            samples, groups = [], []
            for step, gradient in zip(cfg["snapshots"], gradients):
                bodies = np.concatenate([particles(center=center, radius=0.05 + 0.01 * i, time=float(step**2)) for i, center in enumerate(centers[:step])])
                bodies[:, 7:10] = [0.1 * step, 0.2, -0.1]
                # Zero saved net loads model cancellation by an opposing body
                # force; the nonzero pressure load must still be recovered.
                pressure = 10 + sum(g * x for g, x in zip(gradient, coordinates))
                fields = conservative(np.ones(grid.shape), [np.zeros(grid.shape)] * 3, pressure)
                np.concatenate([field.ravel(order="F") for field in fields]).tofile(root / f"restart_data/{step}.dat")
                bodies.tofile(root / f"restart_data/ib_state_{step}.dat")
                force = -(4 * np.pi * bodies[:, 19, None] ** 3 / 3) * gradient
                groups.append(force)
                samples.extend(force)
            samples = np.asarray(samples)
            validate(cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                moments, _ = process_snapshots(cfg, root, range(3), 0)
            for name in HYDRO_PARTICLE_FIELDS:
                self.assertEqual(moments[name, None].count.item(), 6)
                self.assertEqual(moments[name, None].weight.item(), 6)
            outputs = []
            for mode in ("serial", "parallel", "one_filter"):
                cfg["output"] = f"{mode}.npz"
                if mode == "one_filter":
                    cfg["sigmas"] = [0.14]
                    cfg["diagnostics"] = {"fields": ["alpha"], "map_indices": [[1, 1, 1]], "particle_loads": False}
                path = root / "config.json"
                path.write_text(json.dumps(cfg))
                with contextlib.redirect_stdout(io.StringIO()):
                    run(path, snapshot_workers=2 if mode == "parallel" else 1)
                with np.load(root / cfg["output"]) as result:
                    metadata = json.loads(str(result["metadata"]))
                    self.assertEqual(metadata["hydrodynamic_particle_statistics"]["realizations"], 6)
                    self.assertFalse(metadata["hydrodynamic_particle_statistics"]["filter_dependent"])
                    assert_allclose(result["particle_force_x.mean"], 0, atol=0)
                    for column, axis in enumerate("xyz"):
                        values = samples[:, column]
                        centered = values - values.mean()
                        variance = np.mean(centered**2)
                        expected = [values.mean(), variance, np.mean(centered**3) / variance**1.5, np.mean(centered**4) / variance**2]
                        for statistic, value in zip(STATISTICS, expected):
                            key = f"particle_hydro_force_{axis}.{statistic}"
                            self.assertEqual(result[key].shape, ())
                            assert_allclose(result[key], value, rtol=1e-10, atol=1e-14)
                        if mode != "one_filter":
                            prefix = f"diagnostics.particle_hydro_force_{axis}"
                            assert_allclose(result[f"{prefix}.count"][:, 0], [1, 2, 3])
                            assert_allclose(result[f"{prefix}.mean"][:, 0], [group[:, column].mean() for group in groups], atol=1e-14)
                            assert_allclose(result[f"{prefix}.m2"][:, 0], [np.sum((group[:, column] - group[:, column].mean()) ** 2) for group in groups], atol=1e-14)
                    if mode == "one_filter":
                        self.assertFalse(any(key.startswith("diagnostics.particle_") for key in result.files))
                    outputs.append({key: result[key].copy() for key in result.files if key.startswith("particle_hydro_")})
            for other in outputs[1:]:
                self.assertEqual(set(other), set(outputs[0]))
                for key, value in other.items():
                    assert_allclose(value, outputs[0][key], rtol=1e-10, atol=1e-14, equal_nan=True, err_msg=key)


if __name__ == "__main__":
    unittest.main()
