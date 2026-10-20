"""Pooled higher moments against direct sample statistics, including final output."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .__main__ import KEY_FORMAT, PARTICLE_FIELDS, STATISTICS, VERSION, atomic_save, pack, run
from .compact import convert
from .maps import Moments
from .test_volume_filter import config, particles, synthetic_state


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

    def test_unit_updates_after_weighted_history(self):
        samples = np.array([[1.0, np.nan], [5.0, 4.0], [-2.0, 8.0], [np.nan, 3.0], [9.0, 2.0]])
        weights = np.array([[0.2, 0.0], [1.7, 0.8], [1, 1], [1, 1], [1, 1]])
        result = accumulate(samples[:2], weights[:2])
        for sample in samples[2:]:
            result.update(sample)
        self.assert_direct(result, samples, weights)

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

    def test_final_output_roundtrip_and_atomic_failure(self):
        samples = np.array([[1.0, np.nan], [4.0, 3.0], [9.0, 3.0], [-2.0, 3.0]])
        result = accumulate(samples)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.npz"
            metadata = {"version": VERSION, "output_mode": "final_only"}
            atomic_save(path, pack({("F_x", 0): result, ("F_x", 1): result}, metadata))
            with np.load(path) as data:
                self.assertEqual(json.loads(str(data["metadata"])), metadata)
                expected_keys = {"metadata"} | {f"F_x.{name}.sigma{i}" for name in STATISTICS for i in (0, 1)}
                self.assertEqual(set(data.files), expected_keys)
                for name in STATISTICS:
                    for i in (0, 1):
                        assert_allclose(data[f"F_x.{name}.sigma{i}"], getattr(result, name), equal_nan=True)

            def broken_output():
                yield "first", np.ones(2)
                raise RuntimeError("interrupted output")

            failed = Path(directory) / "failed.npz"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                atomic_save(failed, broken_output())
            self.assertFalse(failed.exists())
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_convert_preserves_pooled_values_and_rejects_incomplete_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "old.npz", root / "compact.npz"
            metadata = {"version": 2, "completed": 2, "config": {"snapshots": [1, 2]}, "statistics": {"pooled_weighting": "equal_realizations", "snapshot_weighting": "equal_snapshots"}}
            arrays = {"metadata": np.array(json.dumps(metadata))}
            for index in (0, 1):
                for name in ("F_x", "alpha_R_u_xx"):
                    for j, statistic in enumerate(STATISTICS):
                        value = np.array([index + j, np.nan])
                        arrays[f"sigma{index}.{name}.pooled.{statistic}"] = value
                        arrays[f"sigma{index}.{name}.time.{statistic}"] = value + 100
                    arrays[f"sigma{index}.{name}.pooled.count"] = np.array([3, 0])
                    arrays[f"sigma{index}.{name}.pooled.raw_moment4"] = np.array([999, 0])
            atomic_save(source, arrays)
            convert(source, output)
            with np.load(output) as data:
                expected = {"metadata"}
                for index in (0, 1):
                    for name in ("F_x", "alpha_R_u_xx"):
                        for statistic in STATISTICS:
                            new = f"{name}.{statistic}.sigma{index}"
                            expected.add(new)
                            assert_array_equal(data[new], arrays[f"sigma{index}.{name}.pooled.{statistic}"])
                self.assertEqual(set(data.files), expected)
                self.assertEqual(json.loads(str(data["metadata"]))["version"], VERSION)
            with self.assertRaises(FileExistsError):
                convert(source, output)
            metadata["completed"] = 1
            arrays["metadata"] = np.array(json.dumps(metadata))
            atomic_save(source, arrays)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                convert(source, root / "incomplete.npz")
            self.assertFalse((root / "incomplete.npz").exists())

    def test_pipeline_pools_particles_not_snapshot_means(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = config()
            cfg.update(fields=["F_x"], sigmas=[0.1, 0.2], map={"kind": "box", "length": 0.125, "points": 3})
            times = [0.0, 1.0, 4.0]
            cfg["snapshots"] = [0, 1, 2]
            positions = [[0.28125], [0.28125, 0.40625, 0.65625], [0.40625, 0.78125]]
            expected = []
            for i, centers in enumerate(positions):
                (root / f"restart_data/{i}.dat").write_bytes(b"fixture")
                states = np.concatenate([particles(center=(x, 0.5, 0.5), time=times[i]) for x in centers])
                states.tofile(root / f"restart_data/ib_state_{i}.dat")
                expected.extend([10 * times[i] + x for x in centers])

            def synthetic_fields(grid, state, states, sigma, options):
                x = (np.arange(grid.shape[0]) + 0.5) * grid.dx[0]
                field = np.broadcast_to(x[:, None, None] + 10 * states[0, 0], grid.shape)
                yield "F_x", (1 + sigma) * field + 2 * sigma

            outputs = []
            for mode in ("steps", "legacy"):
                cfg["output"] = f"{mode}.npz"
                if mode == "legacy":
                    cfg.update(snapshots=[{"step": i, "time": 0.001} for i in range(3)], time_weighting="trapezoid", save_snapshot_maps=True)
                path = root / "config.json"
                path.write_text(json.dumps(cfg))
                progress_log = StringIO()
                with (
                    redirect_stdout(progress_log),
                    patch("mfc.volume_filter.__main__.reconstruct", side_effect=lambda *args: synthetic_state()),
                    patch("mfc.volume_filter.__main__.closure_fields", side_effect=synthetic_fields),
                    patch("mfc.volume_filter.__main__.atomic_save", wraps=atomic_save) as save,
                ):
                    run(path, progress_interval=0)
                    self.assertEqual(save.call_count, 1)
                self.assertFalse(list(root.glob("*.step*.npz")))
                log = progress_log.getvalue()
                for i, centers in enumerate(positions):
                    label = f"Snapshot {i + 1}/3 step={i}"
                    self.assertEqual(log.count(f"{label}: reading particle state"), 1)
                    for j, sigma in enumerate(cfg["sigmas"], 1):
                        prefix = f"{label} filter={j}/2 sigma={sigma} F_x: particles"
                        self.assertEqual(log.count(prefix), 1)
                        self.assertIn(f"{prefix} {len(centers)}/{len(centers)}; compute", log)
                    self.assertIn(f"{label}: complete; statistics retained in memory", log)
                self.assertNotIn("full-domain field ready", log)
                self.assertNotIn("(100.0%)", log)
                with np.load(root / cfg["output"]) as data:
                    metadata = json.loads(str(data["metadata"]))
                    self.assertEqual(metadata["snapshot_times"], times)
                    self.assertEqual(metadata["config"]["snapshots"], [0, 1, 2])
                    self.assertEqual(metadata["output_mode"], "final_only")
                    self.assertEqual(metadata["completed"], 3)
                    self.assertEqual(metadata["key_format"], KEY_FORMAT)
                    self.assertEqual(metadata["statistics"]["outputs"], list(STATISTICS))
                    expected_keys = {"metadata"} | {f"F_x.{name}.sigma{i}" for name in STATISTICS for i in (0, 1)}
                    expected_keys |= {f"{field}.{statistic}" for field in PARTICLE_FIELDS for statistic in STATISTICS}
                    self.assertEqual(set(data.files), expected_keys)
                    for i, sigma in enumerate(cfg["sigmas"]):
                        samples = np.broadcast_to(np.array(expected)[:, None, None, None] + np.linspace(-0.0625, 0.0625, 3)[None, :, None, None], (6, 3, 3, 3))
                        samples = (1 + sigma) * samples + 2 * sigma
                        reference = accumulate(samples)
                        self.assert_direct(reference, samples)
                        for name in STATISTICS:
                            assert_allclose(data[f"F_x.{name}.sigma{i}"], getattr(reference, name), atol=1e-12)
                        snapshot_mean = np.mean([10 * t + np.mean(c) for t, c in zip(times, positions)]) * (1 + sigma) + 2 * sigma
                        self.assertNotAlmostEqual(data[f"F_x.mean.sigma{i}"][1, 1, 1], snapshot_mean)
                    outputs.append({key: data[key].copy() for key in data.files if key != "metadata"})
            for key in outputs[0]:
                assert_allclose(outputs[0][key], outputs[1][key])


if __name__ == "__main__":
    unittest.main()
