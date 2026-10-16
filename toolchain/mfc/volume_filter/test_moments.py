"""Pooled higher moments against direct sample statistics, including resume."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .__main__ import VERSION, atomic_save, pack, restore, run
from .maps import Moments
from .test_volume_filter import config, particles


def accumulate(samples, weights=None):
    result = Moments(samples.shape[1:])
    for i, sample in enumerate(samples):
        result.update(sample, 1 if weights is None else weights[i])
    return result


class MomentTests(unittest.TestCase):
    def assert_direct(self, result, samples, weights=None, rtol=1e-12):
        if weights is None:
            weights = np.ones_like(samples)
        weights = np.broadcast_to(weights, samples.shape)
        for index in np.ndindex(result.mean.shape):
            values = samples[(slice(None), *index)]
            w = weights[(slice(None), *index)]
            good = np.isfinite(values) & (w > 0)
            values, w = values[good].astype(np.longdouble), w[good].astype(np.longdouble)
            self.assertEqual(result.count[index], len(values))
            self.assertAlmostEqual(result.weight[index], float(w.sum()))
            if not len(values):
                self.assertTrue(np.isnan(result.raw_moment(1)[index]))
                for order in (2, 3, 4):
                    self.assertTrue(np.isnan(result.central_moment(order)[index]))
                continue
            mean = np.sum(w * values) / w.sum()
            assert_allclose(result.mean[index], float(mean), rtol=rtol, atol=1e-13)
            central = {}
            for order in (2, 3, 4):
                central[order] = np.sum(w * (values - mean) ** order) / w.sum()
                assert_allclose(result.central_moment(order)[index], float(central[order]), rtol=rtol, atol=rtol)
            for order in (1, 2, 3, 4):
                raw = np.sum(w * values**order) / w.sum()
                assert_allclose(result.raw_moment(order)[index], float(raw), rtol=rtol, atol=rtol)
            if central[2] > 0:
                assert_allclose(result.skewness[index], float(central[3] / central[2] ** 1.5), rtol=rtol, atol=rtol)
                assert_allclose(result.kurtosis[index], float(central[4] / central[2] ** 2), rtol=rtol, atol=rtol)
            else:
                self.assertTrue(np.isnan(result.skewness[index]))
                self.assertTrue(np.isnan(result.kurtosis[index]))

    def test_asymmetric_stream_and_unequal_groups_match_direct(self):
        rng = np.random.default_rng(81)
        samples = rng.exponential(2, size=(43, 2, 3)) - 0.5
        samples[::3, 0, 0] = np.nan
        samples[:, 1, 2] = np.nan
        sequential = accumulate(samples)
        self.assert_direct(sequential, samples)
        groups = [accumulate(part) for part in np.split(samples, [1, 4, 12, 42])]
        for order in ([0, 1, 2, 3, 4], [4, 2, 0, 3, 1]):
            merged = Moments((2, 3))
            for i in order:
                merged.merge(groups[i])
            self.assert_direct(merged, samples)
            assert_array_equal(merged.count, sequential.count)

    def test_weighted_diagnostics_keep_actual_count(self):
        samples = np.array([[1.0, 2.0], [2.0, 5.0], [8.0, np.nan], [-4.0, 7.0]])
        weights = np.array([[0.0, 2.0], [0.5, 3.0], [1.5, 1.0], [4.0, 1.0]])
        result = accumulate(samples, weights)
        self.assert_direct(result, samples, weights)
        merged = accumulate(samples[:2], weights[:2])
        merged.merge(accumulate(samples[2:], weights[2:]))
        self.assert_direct(merged, samples, weights)

    def test_empty_constant_singleton_and_population_normalization(self):
        samples = np.array([[np.nan, 3.0, np.nan, 0.0], [np.nan, 3.0, 7.0, 4.0]])
        result = accumulate(samples)
        self.assert_direct(result, samples)
        assert_allclose(result.variance[3], 4)  # Population normalization, not N-1.
        assert_allclose(result.kurtosis[3], 1)
        empty = Moments((4,))
        result.merge(empty)
        self.assert_direct(result, samples)
        empty.merge(result)
        self.assert_direct(empty, samples)
        for weight in (-1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "weights"):
                result.update(np.ones(4), weight)

    def test_large_offset_small_fluctuations(self):
        samples = 1e9 + np.array([0, 1, 1, 2, 2, 2, 4, 6, 10, 12], dtype=float)[:, None] / 8
        self.assert_direct(accumulate(samples), samples, rtol=2e-6)
        merged = accumulate(samples[:3])
        merged.merge(accumulate(samples[3:]))
        self.assert_direct(merged, samples, rtol=2e-6)

    def test_checkpoint_roundtrip_and_legacy_rejection(self):
        samples = np.array([[1.0, np.nan], [4.0, 3.0], [9.0, 3.0], [-2.0, 3.0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            first = accumulate(samples[:2])
            metadata = {"version": VERSION, "config_hash": "test"}
            atomic_save(path, pack({"sigma0.F_x.pooled": first}, metadata))
            loaded, _ = restore(path, "test")
            loaded["sigma0.F_x.pooled"].merge(accumulate(samples[2:]))
            self.assert_direct(loaded["sigma0.F_x.pooled"], samples)
            metadata["version"] = 1
            atomic_save(path, pack({"sigma0.F_x.pooled": first}, metadata))
            with self.assertRaisesRegex(ValueError, "reprocessing"):
                restore(path, "test")

    def test_pipeline_pools_particles_not_snapshot_means(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = config()
            cfg.update(fields=["F_x"], map={"kind": "box", "x": [0], "y": [0], "z": [0]})
            cfg["snapshots"] = [{"step": i, "time": time} for i, time in enumerate([0.0, 1.0, 4.0])]
            positions = [[0.28125], [0.28125, 0.40625, 0.65625], [0.40625, 0.78125]]
            expected = []
            for i, centers in enumerate(positions):
                (root / f"restart_data/{i}.dat").write_bytes(b"fixture")
                states = np.concatenate([particles(center=(x, 0.5, 0.5), time=cfg["snapshots"][i]["time"]) for x in centers])
                states.tofile(root / f"restart_data/ib_state_{i}.dat")
                expected.extend([10 * cfg["snapshots"][i]["time"] + x for x in centers])

            def synthetic_fields(grid, state, states, sigma, options):
                x = (np.arange(grid.shape[0]) + 0.5) * grid.dx[0]
                field = np.broadcast_to(x[:, None, None] + 10 * states[0, 0], grid.shape)
                yield "F_x", field

            outputs = []
            for mode in ("sample", "trapezoid"):
                cfg.update(time_weighting=mode, output=f"{mode}.npz")
                path = root / "config.json"
                path.write_text(json.dumps(cfg))
                with patch("mfc.volume_filter.__main__.reconstruct", return_value=None), patch("mfc.volume_filter.__main__.closure_fields", side_effect=synthetic_fields):
                    run(path, max_snapshots=1)
                    run(path, resume=True)
                with np.load(root / cfg["output"]) as data:
                    prefix = "sigma0.F_x.pooled"
                    self.assertEqual(data[prefix + ".count"].item(), 6)
                    reference = accumulate(np.array(expected).reshape(-1, 1, 1, 1))
                    self.assert_direct(reference, np.array(expected).reshape(-1, 1, 1, 1))
                    for name in ("mean", "variance", "skewness", "kurtosis"):
                        assert_allclose(data[prefix + "." + name], getattr(reference, name), atol=1e-12)
                    for order in (1, 2, 3, 4):
                        assert_allclose(data[prefix + f".raw_moment{order}"], np.mean(np.array(expected) ** order))
                    outputs.append({key: data[key].copy() for key in data.files if ".pooled." in key})
                    self.assertNotAlmostEqual(data[prefix + ".variance"].item(), data["sigma0.F_x.time.variance"].item())
            for key in outputs[0]:
                assert_allclose(outputs[0][key], outputs[1][key])


if __name__ == "__main__":
    unittest.main()
