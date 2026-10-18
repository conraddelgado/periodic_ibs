"""Real spawned snapshot processes against serial processing and failure cleanup."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from .__main__ import run
from .numerics import Grid
from .test_volume_filter import config, conservative, particles


class ParallelSnapshotTests(unittest.TestCase):
    def test_workers_match_serial_and_preserve_snapshot_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = config()
            cfg.update(snapshots=[1, 2, 3], fft_workers=2, fields=[], sigmas=[0.1, 0.15])
            cfg["physics"] = dict(cfg["physics"], mu=0.01, conductivity=0.02)
            cfg["surface"] = dict(cfg["surface"], thermal_bc="reconstruct")
            grid = Grid(**cfg["grid"])
            x = (np.arange(grid.shape[0]) + 0.5)[:, None, None] / grid.shape[0]
            centers = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75), (0.25, 0.75, 0.25)]
            for step in cfg["snapshots"]:
                rho = np.broadcast_to(1 + 0.05 * np.sin(2 * np.pi * x), grid.shape).copy()
                velocity = [np.broadcast_to(0.2 * step + 0.05 * np.cos(2 * np.pi * x), grid.shape), np.zeros(grid.shape), np.zeros(grid.shape)]
                fields = conservative(rho, velocity, np.full(grid.shape, 1.0 + step))
                np.concatenate([field.ravel(order="F") for field in fields]).tofile(root / f"restart_data/{step}.dat")
                np.concatenate([particles(center=center, radius=0.08, time=step / 10) for center in centers[:step]]).tofile(root / f"restart_data/ib_state_{step}.dat")
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            run(path)
            cfg["output"] = "parallel.npz"
            path.write_text(json.dumps(cfg))
            run(path, snapshot_workers=2)
            with np.load(root / "stats.npz") as serial, np.load(root / "parallel.npz") as parallel:
                self.assertEqual(set(serial.files), set(parallel.files))
                for key in serial.files:
                    if key != "metadata":
                        assert_allclose(serial[key], parallel[key], rtol=1e-8, atol=1e-10, equal_nan=True, err_msg=key)
                metadata = json.loads(str(parallel["metadata"]))
                self.assertEqual(metadata["completed"], 3)
                self.assertEqual(metadata["particle_counts"], [1, 2, 3])
                self.assertEqual(metadata["snapshot_times"], [0.1, 0.2, 0.3])
                self.assertEqual(metadata["execution"], {"snapshot_workers": 2, "fft_workers_per_worker": 1})
                self.assertEqual([entry[0][0] for entry in metadata["inputs"]], [str(root / f"restart_data/{step}.dat") for step in (1, 2, 3)])
            self.assertEqual(len(list(root.glob("*.npz"))), 2)
            cfg.update(output="failed.npz", snapshots=[1, 99])
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(RuntimeError, "worker.*failed"):
                run(path, snapshot_workers=2)
            self.assertFalse((root / "failed.npz").exists())
            self.assertFalse(list(root.glob("*.tmp")))

    def test_invalid_worker_budget_fails_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            cfg = config()
            cfg["fft_workers"] = 1
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                run(path, snapshot_workers=0)
            with self.assertRaisesRegex(ValueError, "total CPU budget"):
                run(path, snapshot_workers=2)
            self.assertFalse(list(path.parent.glob("*.npz")))


if __name__ == "__main__":
    unittest.main()
