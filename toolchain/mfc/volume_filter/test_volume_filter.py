"""Analytic identities, native binary layouts, particle maps, and restart tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.io import FortranFile

from .__main__ import run, validate
from .closures import closure_fields, reconstruct
from .maps import Moments, ParticleMap
from .numerics import Gaussian, Grid, fluid_derivative, sphere_mask
from .reader import Reader

PHYSICS = {"eos": "ideal_gas", "transport": "constant", "gamma": 1.4, "gas_constant": 1.0, "mu": 0.0, "conductivity": 0.0}
SURFACE = {"polar_points": 6, "offset_cells": 1.5, "thermal_bc": "adiabatic", "velocity_bc": "no_slip"}


def particles(center=(0.5, 0.5, 0.5), radius=0.12, time=0):
    state = np.zeros((1, 20))
    state[0, 0] = time
    state[0, 16:19] = center
    state[0, 19] = radius
    return state


def conservative(rho, u, p):
    energy = p / (PHYSICS["gamma"] - 1) + 0.5 * rho * sum(v**2 for v in u)
    return [rho.copy(), *(rho * v for v in u), energy]


def config():
    return {
        "periodic": [True, True, True],
        "grid": {"shape": [16, 16, 16], "lengths": [1, 1, 1]},
        "physics": PHYSICS,
        "surface": SURFACE,
        "io": {
            "format": "shared",
            "dtype": "<f8",
            "nvars": 5,
            "indices": {"rho": 1, "momentum": [2, 3, 4], "energy": 5},
            "flow": "restart_data/{step}.dat",
            "particles": "restart_data/ib_state_{step}.dat",
        },
        "sigmas": [0.1],
        "map": {"kind": "box", "x": [-0.1, 0, 0.1], "y": [0], "z": [0]},
        "snapshots": [1, 2],
        "output": "stats.npz",
        "fields": ["alpha", "R_u_xx", "F_x"],
        "save_snapshot_maps": True,
    }


class NumericalTests(unittest.TestCase):
    def test_fourier_gaussian_and_derivative(self):
        grid = Grid((16, 18, 20), (2, 3, 4))
        x = (np.arange(16) + 0.5) * grid.dx[0]
        k = 2 * np.pi / 2
        wave = np.broadcast_to(np.sin(k * x)[:, None, None], grid.shape)
        kernel = Gaussian(grid, 0.13)
        assert_allclose(kernel(np.ones(grid.shape)), 1, atol=2e-15)
        assert_allclose(kernel(wave), np.exp(-0.5 * (0.13 * k) ** 2) * wave, atol=2e-15)
        assert_allclose(kernel.derivative(wave, 0), np.broadcast_to((k * np.cos(k * x))[:, None, None], grid.shape), atol=2e-14)
        self.assertAlmostEqual(float(kernel(wave).sum()), float(wave.sum()), places=11)

    def test_periodic_sphere_and_sampling(self):
        grid = Grid((20, 20, 20), (1, 1, 1))
        a = sphere_mask(grid, particles((0.98, 0.5, 0.5)))
        b = sphere_mask(grid, particles((-0.02, 0.5, 0.5)))
        self.assertTrue(np.array_equal(a, b))
        rng = np.random.default_rng(13)
        field = rng.normal(size=grid.shape)
        points = rng.uniform(-2, 2, size=(80, 3))
        assert_allclose(grid.sample(field, points), grid.sample(field, points + [1, -2, 3]), atol=1e-13)
        source = grid.deposit(points, np.arange(80))
        self.assertAlmostEqual(source.sum() * grid.dv, np.arange(80).sum(), places=10)
        # An invalid zero-weight neighbor must not invalidate an exact cell center.
        field[0, 0, 0] = np.nan
        self.assertTrue(np.isfinite(grid.sample(field, [[0.075, 0.025, 0.025]])[0]))

    def test_fluid_derivatives_do_not_read_solid(self):
        grid = Grid((20, 8, 8), (1, 1, 1))
        x = (np.arange(20) + 0.5) / 20
        fluid = np.zeros(grid.shape, dtype=bool)
        fluid[4:15] = True
        field = np.broadcast_to(x[:, None, None] ** 2, grid.shape).copy()
        field[~fluid] = 1e12
        derivative = fluid_derivative(field, fluid, 0, grid.dx[0])
        expected = np.broadcast_to(2 * x[:, None, None], grid.shape)
        assert_allclose(derivative[fluid], expected[fluid], atol=1e-13)
        fluid[:] = False
        fluid[5] = True
        with self.assertRaisesRegex(ValueError, "stencil"):
            fluid_derivative(field, fluid, 0, grid.dx[0])

    def test_analytic_momentum_and_energy_closures(self):
        grid = Grid((20, 8, 8), (1, 1, 1))
        x = (np.arange(20) + 0.5) / 20
        u = np.broadcast_to(np.sin(2 * np.pi * x)[:, None, None], grid.shape).copy()
        rho, p = np.full(grid.shape, 2.0), np.full(grid.shape, 5.0)
        uvec = [u, np.zeros(grid.shape), np.zeros(grid.shape)]
        state = reconstruct(grid, iter(conservative(rho, uvec, p)), np.empty((0, 20)), PHYSICS)
        sigma = 0.08
        result = dict(closure_fields(grid, state, np.empty((0, 20)), sigma, {"surface": SURFACE}))
        ksigma2 = (2 * np.pi * sigma) ** 2
        expected = rho * (0.5 * (1 - np.exp(-2 * ksigma2) * np.cos(4 * np.pi * x)[:, None, None]) - np.exp(-ksigma2) * u**2)
        assert_allclose(result["R_u_xx"], expected, atol=2e-15)
        g = Gaussian(grid, sigma)
        energy = conservative(rho, uvec, p)[-1]
        exact_flux = g((energy + p) * u) - (g(energy) + g(p)) * g(u)
        assert_allclose(result["R_uu_x"] + result["R_Tu_x"], exact_flux, atol=1e-14)
        pressure_recovered = (PHYSICS["gamma"] - 1) * (g(energy) - 0.5 * rho * g(u) ** 2 - 0.5 * result["R_u_xx"])
        assert_allclose(pressure_recovered, p, atol=2e-15)

    def test_viscous_and_thermal_closures(self):
        grid = Grid((24, 8, 8), (1, 1, 1))
        x = (np.arange(24) + 0.5) / 24
        k, sigma = 2 * np.pi, 0.08
        sine = np.broadcast_to(np.sin(k * x)[:, None, None], grid.shape)
        cosine = np.broadcast_to(np.cos(k * x)[:, None, None], grid.shape)
        rho = np.ones(grid.shape)
        p = 5 + 0.2 * cosine
        velocity = [sine, np.zeros(grid.shape), np.zeros(grid.shape)]
        physics = dict(PHYSICS, mu=0.3, conductivity=0.7)
        state = reconstruct(grid, iter(conservative(rho, velocity, p)), np.empty((0, 20)), physics)
        result = dict(closure_fields(grid, state, np.empty((0, 20)), sigma, {"surface": SURFACE}))
        g = Gaussian(grid, sigma)
        attenuation = np.exp(-0.5 * (k * sigma) ** 2)
        discrete_k = np.sin(k * grid.dx[0]) / grid.dx[0]
        tau = (4 / 3) * physics["mu"] * discrete_k * cosine
        expected_stress = (4 / 3) * physics["mu"] * attenuation * (discrete_k - k) * cosine
        assert_allclose(result["R_mu_xx"], expected_stress, atol=1e-14)
        assert_allclose(result["R_mu_yy"], -expected_stress / 2, atol=1e-14)
        assert_allclose(result["R_tau_u_x"], g(sine * tau) - g(sine) * g(tau), atol=1e-14)
        expected_heat = 0.2 * physics["conductivity"] * attenuation * (discrete_k - k) * sine
        assert_allclose(result["R_q_x"], expected_heat, atol=2e-14)

    def test_interfacial_heat_and_viscous_power(self):
        grid = Grid((24, 24, 24), (1, 1, 1))
        axes = [(np.arange(n) + 0.5) * h - 0.5 for n, h in zip(grid.shape, grid.dx)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        # Locally polynomial traces are exact for the chosen centered/one-sided
        # differences; the discontinuity of their periodic extension is far away.
        rho = np.ones(grid.shape)
        temperature = 4 + x * x + y * y + z * z
        velocity = [x * y, np.zeros(grid.shape), np.zeros(grid.shape)]
        physics = dict(PHYSICS, mu=0.2, conductivity=0.3)
        part = particles()
        part[0, 10:13] = [0, 0, 0.8]
        state = reconstruct(grid, iter(conservative(rho, velocity, temperature)), part, physics)
        surface = dict(SURFACE, thermal_bc="reconstruct")
        result = dict(closure_fields(grid, state, part, 0.08, {"surface": surface}))
        volume = 4 * np.pi * part[0, 19] ** 3 / 3
        # q=-2k*r: integral q.n=-6k*volume. tau from u=(xy,0,0)
        # has div(tau)=(0,mu/3,0), and zero integrated torque here.
        self.assertAlmostEqual(result["Q"].sum() * grid.dv, -6 * physics["conductivity"] * volume, places=12)
        self.assertAlmostEqual(result["F_y"].sum() * grid.dv, -physics["mu"] * volume / 3, places=12)
        self.assertAlmostEqual(result["F_u"].sum() * grid.dv, 0, places=12)

    def test_surface_force_power_integrals_and_translation(self):
        grid = Grid((24, 24, 24), (1, 1, 1))
        x = (np.arange(24) + 0.5) / 24
        p = np.broadcast_to(3 + 0.4 * x[:, None, None], grid.shape).copy()
        rho = np.ones(grid.shape)
        velocity = [np.full(grid.shape, 0.7), np.zeros(grid.shape), np.zeros(grid.shape)]
        part = particles()
        part[0, 7] = 0.7
        part[0, 10:13] = [0.2, -0.3, 0.4]
        state = reconstruct(grid, iter(conservative(rho, velocity, p)), part, PHYSICS)
        result = dict(closure_fields(grid, state, part, 0.08, {"surface": SURFACE}))
        force = 0.4 * 4 * np.pi * part[0, 19] ** 3 / 3
        self.assertAlmostEqual(result["F_x"].sum() * grid.dv, force, places=13)
        self.assertAlmostEqual(result["F_y"].sum() * grid.dv, 0, places=13)
        self.assertAlmostEqual(result["F_u"].sum() * grid.dv, 0.7 * force, places=13)
        assert_allclose(result["R_u_xx"], 0, atol=1e-14)
        assert_allclose(result["Q"], 0, atol=0)
        shifted = part.copy()
        shifted[:, 16] += grid.dx[0] * 3
        constant = conservative(rho, velocity, np.ones(grid.shape))
        base = dict(closure_fields(grid, reconstruct(grid, iter(constant), part, PHYSICS), part, 0.08, {"surface": SURFACE}))
        constant = conservative(rho, velocity, np.ones(grid.shape))
        moved = dict(closure_fields(grid, reconstruct(grid, iter(constant), shifted, PHYSICS), shifted, 0.08, {"surface": SURFACE}))
        assert_allclose(moved["F_x"], np.roll(base["F_x"], 3, axis=0), atol=1e-13)

    def test_axisymmetric_map_and_moments(self):
        grid = Grid((20, 20, 20), (1, 1, 1))
        mapper = ParticleMap(grid, {"kind": "axisymmetric", "axial": [-0.1, 0.1], "radial": [0, 0.1], "azimuthal_points": 16})
        result = mapper.particles(np.full(grid.shape, 4), particles())
        assert_allclose(result.mean, 4)
        moments = Moments((2,))
        moments.update(np.array([1.0, np.nan]), 2)
        moments.update(np.array([3.0, 8.0]), 1)
        assert_allclose(moments.mean, [5 / 3, 8])
        assert_allclose(moments.variance, [8 / 9, 0])
        pooled = Moments((1,))
        for group in ([1.0, 3.0], [5.0, 7.0]):
            batch = Moments((1,))
            for value in group:
                batch.update(np.array([value]))
            pooled.merge(batch)
        assert_allclose(pooled.mean, [4])
        assert_allclose(pooled.variance, [5])


class ReaderAndPipelineTests(unittest.TestCase):
    def test_native_shared_shards_serial_and_particles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid = Grid((8, 6, 6), (1, 1, 1))
            fields = [np.arange(np.prod(grid.shape)).reshape(grid.shape) + 1000 * i for i in range(5)]
            shared = np.concatenate([f.ravel(order="F") for f in fields]).astype("<f8")
            shared.tofile(root / "1.dat")
            part = particles(time=0.4)
            part.astype("<f8").tofile(root / "ib1.dat")
            settings = {"format": "shared", "nvars": 5, "flow": "{step}.dat", "particles": "ib{step}.dat", "indices": {"rho": 1, "momentum": [2, 3, 4], "energy": 5}}
            reader = Reader(settings, grid, root)
            for actual, expected in zip(reader.fields(1), fields):
                assert_allclose(actual, expected)
            assert_allclose(reader.particles(1), part)
            blocks = [{"offset": [4 * rank, 0, 0], "shape": [4, 6, 6], "rank": rank} for rank in range(2)]
            for rank in range(2):
                local = [f[4 * rank : 4 * (rank + 1)] for f in fields]
                np.concatenate([f.ravel(order="F") for f in local]).astype("<f4").tofile(root / f"1_{rank}.dat")
                for var, field in enumerate(local, 1):
                    with FortranFile(root / f"1_{rank}_{var}.dat", "w") as stream:
                        stream.write_record(field.ravel(order="F").astype("<f4"))
                with open(root / f"ib1_{rank}.dat", "wb") as stream:
                    np.array([1], dtype="<i4").tofile(stream)
                    np.array([1], dtype="<i4").tofile(stream)
                    part.astype("<f8").tofile(stream)
            settings.update(format="shards", dtype="<f4", blocks=blocks, flow="{step}_{rank}.dat", particle_shards=True, particles="ib{step}_{rank}.dat")
            for mode, path in [("shards", "{step}_{rank}.dat"), ("serial", "{step}_{rank}_{var}.dat")]:
                settings.update(format=mode, flow=path)
                reader = Reader(settings, grid, root)
                for actual, expected in zip(reader.fields(1), fields):
                    assert_allclose(actual, expected)
                assert_allclose(reader.particles(1), part)
            settings["blocks"][1]["offset"] = [3, 0, 0]
            with self.assertRaisesRegex(ValueError, "Overlapping"):
                Reader(settings, grid, root)

    def test_end_to_end_resume_equals_single_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = config()
            grid = Grid(**cfg["grid"])
            for step in (1, 2):
                rho, p = np.ones(grid.shape), np.full(grid.shape, 1.0 + step)
                fields = conservative(rho, [np.zeros(grid.shape)] * 3, p)
                np.concatenate([f.ravel(order="F") for f in fields]).tofile(root / f"restart_data/{step}.dat")
                particles(time=float(step)).tofile(root / f"restart_data/ib_state_{step}.dat")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg))
            run(config_path, max_snapshots=1)
            run(config_path, resume=True)
            with np.load(root / "stats.npz") as data:
                resumed = {key: data[key].copy() for key in data.files if key != "metadata"}
                metadata = json.loads(str(data["metadata"]))
                self.assertEqual(metadata["completed"], 2)
                self.assertEqual(metadata["snapshot_times"], [1.0, 2.0])
            cfg["output"] = "single.npz"
            config_path.write_text(json.dumps(cfg))
            run(config_path)
            with np.load(root / "single.npz") as data:
                for key, value in resumed.items():
                    assert_allclose(data[key], value, equal_nan=True)
            with self.assertRaises(FileExistsError):
                run(config_path)
            cfg["sigmas"] = [0.15]
            config_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run(config_path, resume=True)

    def test_snapshot_steps_and_unsupported_physics(self):
        for snapshots in ([0, 1, 4], [{"step": 0}, {"step": 1}, {"step": 4}], [{"step": i, "time": "ignored"} for i in (0, 1, 4)]):
            cfg = config()
            cfg.update(snapshots=snapshots, time_weighting="trapezoid")
            validate(cfg)
            self.assertEqual(cfg["snapshots"], [0, 1, 4])
            self.assertNotIn("time_weighting", cfg)
        for snapshots in ([], [1, 1], [True], [-1], [1.5], [{}], ["1"]):
            cfg = config()
            cfg["snapshots"] = snapshots
            with self.assertRaises(ValueError):
                validate(cfg)
        cfg = config()
        cfg["periodic"] = [True, False, True]
        with self.assertRaisesRegex(ValueError, "periodic"):
            validate(cfg)


if __name__ == "__main__":
    unittest.main()
