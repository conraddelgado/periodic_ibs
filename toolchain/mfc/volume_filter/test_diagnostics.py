"""History extraction, estimator influences, correlated errors, and worker ordering."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from .__main__ import atomic_save, run, validate
from .diagnostics import ARRAYS, capture, history_arrays
from .maps import Moments
from .sampling import (
    analyze_histories,
    autocorrelation,
    block_jackknife,
    correlation_factor,
    influence_series,
    json_safe,
    load_histories,
    main,
    merge_rows,
    mser_cutoff,
    regular_spacing,
)
from .test_volume_filter import config, particles


def history_from_groups(groups):
    rows = []
    for group in groups:
        value = Moments(())
        for sample in group:
            value.update(sample)
        rows.append(capture(value))
    return {name: np.stack([row[name] for row in rows]) for name in ARRAYS}


def direct_statistics(values, weights=None):
    weights = np.ones(len(values)) if weights is None else weights
    mean = np.average(values, weights=weights)
    centered = values - mean
    variance = np.average(centered**2, weights=weights)
    return {
        "mean": mean,
        "variance": variance,
        "skewness": np.average(centered**3, weights=weights) / variance**1.5,
        "kurtosis": np.average(centered**4, weights=weights) / variance**2,
    }


class DiagnosticsTests(unittest.TestCase):
    def test_validation_and_copy_before_merge(self):
        cfg = config()
        validate(cfg)
        self.assertNotIn("diagnostics", cfg)
        cfg["diagnostics"] = {"fields": ["F_x"], "map_indices": [[1, 1, 1]]}
        validate(cfg)
        self.assertTrue(cfg["diagnostics"]["particle_loads"])
        for options in (
            {"fields": ["F_x"], "map_indices": [[3, 1, 1]]},
            {"fields": ["F_x"], "map_indices": [[True, 1, 1]]},
            {"fields": ["F_x"], "map_indices": [[1, 1, 1], [1, 1, 1]]},
            {"fields": ["Q"], "map_indices": [[1, 1, 1]]},
            {"fields": ["F_x"]},
            {"particle_loads": False},
            {"particle_loads": "yes"},
            {"unknown": True},
        ):
            cfg["diagnostics"] = options
            with self.assertRaises(ValueError):
                validate(cfg)
        value = Moments((3, 3, 3))
        value.update(np.ones(value.mean.shape))
        saved = capture(value, [[1, 1, 1]])
        value.update(np.full(value.mean.shape, 9))
        assert_array_equal(saved["count"], [1])
        assert_array_equal(saved["mean"], [1])
        assert_array_equal(saved["m2"], [0])

    def test_native_snapshot_history_and_main_statistics_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "restart_data").mkdir()
            cfg = config()
            cfg.update(snapshots=[1, 2, 3], fields=["F_x"], sigmas=[0.1, 0.2])
            for step in cfg["snapshots"]:
                (root / f"restart_data/{step}.dat").write_bytes(b"fixture")
                states = np.concatenate([particles(center=(0.25, 0.5, 0.5), time=step / 10), particles(center=(0.75, 0.5, 0.5), time=step / 10)])
                states[:, 1] = [step, 3 * step]
                states.tofile(root / f"restart_data/ib_state_{step}.dat")

            def fields(grid, state, states, sigma, options):
                x = (np.arange(grid.shape[0]) + 0.5) * grid.dx[0]
                yield "F_x", np.broadcast_to(x[:, None, None] + 10 * states[0, 0] + sigma, grid.shape)

            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            with patch("mfc.volume_filter.__main__.reconstruct", return_value=None), patch("mfc.volume_filter.__main__.closure_fields", side_effect=fields):
                run(path)
                cfg["diagnostics"] = {"fields": ["F_x"], "map_indices": [[1, 1, 1], [2, 1, 1]]}
                cfg["output"] = "history.npz"
                path.write_text(json.dumps(cfg))
                run(path)
            with np.load(root / "stats.npz") as original, np.load(root / "history.npz") as result:
                for key in original.files:
                    if key != "metadata":
                        assert_allclose(result[key], original[key], equal_nan=True)
                assert_allclose(result["diagnostics.times"], [0.1, 0.2, 0.3])
                assert_array_equal(result["diagnostics.steps"], [1, 2, 3])
                assert_array_equal(result["diagnostics.F_x.sigma0.count"], np.full((3, 2), 2))
                assert_allclose(result["diagnostics.F_x.sigma0.mean"], np.array([1.6, 2.6, 3.6])[:, None] + np.array([0, 0.1])[None, :])
                assert_allclose(result["diagnostics.F_x.sigma1.mean"], result["diagnostics.F_x.sigma0.mean"] + 0.1)
                assert_allclose(result["diagnostics.F_x.sigma0.m2"], 0.125)
                assert_allclose(result["diagnostics.particle_force_x.mean"][:, 0], [2, 4, 6])
                assert_allclose(result["diagnostics.particle_force_x.m2"][:, 0], [2, 8, 18])
                metadata = json.loads(str(result["metadata"]))
                self.assertEqual(metadata["diagnostics"]["version"], 1)
            times, histories = load_histories(root / "history.npz", ["F_x.sigma0"])
            self.assertEqual(set(histories), {"F_x.sigma0"})
            assert_allclose(times, [0.1, 0.2, 0.3])
            with self.assertRaisesRegex(ValueError, "no diagnostics"):
                load_histories(root / "stats.npz")
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_histories(root / "history.npz", ["missing"])

    def test_empty_support_survives_and_nonchronological_history_is_rejected(self):
        value = Moments((2,))
        value.update(np.array([np.nan, 3.0]))
        history = capture(value, [[0], [1]])
        records = [{"diagnostics": {"X.sigma0": history}, "time": 1.0, "step": 3}]
        arrays = dict(history_arrays(records))
        assert_array_equal(arrays["diagnostics.X.sigma0.count"], [[0, 1]])
        records.append(dict(records[0], time=0.5, step=4))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            dict(history_arrays(records))


class SamplingTests(unittest.TestCase):
    def test_fft_matches_direct_covariance_and_ar1_integral_scale(self):
        rng = np.random.default_rng(79)
        values = rng.normal(size=128)
        centered = values - values.mean()
        expected = [np.dot(centered[: len(values) - lag], centered[lag:]) / ((len(values) - lag) * np.mean(centered**2)) for lag in range(17)]
        assert_allclose(autocorrelation(values, 16), expected, atol=1e-14)
        self.assertTrue(np.all(np.isnan(autocorrelation(np.ones(64), 8))))
        phi = 0.75
        innovations = rng.normal(size=34000)
        correlated = np.zeros(len(innovations))
        for index in range(1, len(correlated)):
            correlated[index] = phi * correlated[index - 1] + innovations[index]
        correlated = correlated[1000:]
        factor = correlation_factor(autocorrelation(correlated, 128), len(correlated))
        self.assertAlmostEqual(factor / ((1 + phi) / (1 - phi)), 1, delta=0.3)

    def test_ratio_and_higher_moment_influences_match_weight_perturbations(self):
        groups = [np.array([-2.0]), np.array([1.0, 3.0, 8.0]), np.array([-1.0, 4.0]), np.array([])]
        history = history_from_groups(groups)
        total = merge_rows(history, range(len(groups)))
        influences = influence_series(history, total)
        values = np.concatenate(groups)
        expected = direct_statistics(values)
        epsilon = 1e-6
        for group_index, group in enumerate(groups):
            weights = np.ones(len(values))
            first = sum(len(item) for item in groups[:group_index])
            weights[first : first + len(group)] += epsilon
            changed = direct_statistics(values, weights)
            for name in expected:
                derivative = (changed[name] - expected[name]) * len(groups) / epsilon
                assert_allclose(influences[name][group_index, 0], derivative, rtol=2e-5, atol=1e-6, err_msg=name)

    def test_block_jackknife_recomputes_full_particle_distributions(self):
        groups = [np.array([index - 1.0, index**2 + 4.0, -index / 3])[: index % 3 + 1] for index in range(16)]
        history = history_from_groups(groups)
        result = block_jackknife(history, 2)
        expected = direct_statistics(np.concatenate(groups))
        deleted = [direct_statistics(np.concatenate(groups[:first] + groups[first + 2 :])) for first in range(0, 16, 2)]
        for name, estimate in expected.items():
            assert_allclose(result["estimates"][name], [estimate], rtol=1e-12)
            values = np.array([item[name] for item in deleted])
            error = np.sqrt(7 / 8 * np.sum((values - values.mean()) ** 2))
            assert_allclose(result["standard_errors"][name], [error], rtol=1e-12)

    def test_physical_time_checks_mser_and_insufficient_history(self):
        assert_allclose(regular_spacing(np.arange(64) * 1e-9), 1e-9)
        for times in (np.arange(16), np.zeros(64), np.arange(64)[::-1], np.r_[np.arange(63), 100], np.r_[np.arange(63), np.nan]):
            with self.assertRaises(ValueError):
                regular_spacing(times)
        rng = np.random.default_rng(9)
        values = np.r_[np.linspace(30, 0, 40), rng.normal(scale=0.1, size=160)]
        cutoff = mser_cutoff(values)
        self.assertGreater(cutoff, 0)
        self.assertLessEqual(cutoff, 100)
        history = history_from_groups([[value] for value in values[:64]])
        with self.assertRaisesRegex(ValueError, "Retain"):
            analyze_histories(np.arange(64), {"X": history}, start_time=50)
        with self.assertRaisesRegex(ValueError, "max_lag"):
            analyze_histories(np.arange(64), {"X": history}, start_time=0, max_lag=40)

    def test_zero_mean_requires_absolute_tolerance_and_oscillations_are_retained(self):
        groups = [[-1.0], [1.0]] * 48
        history = history_from_groups(groups)
        report = analyze_histories(np.arange(96), {"zero": history}, start_time=0)
        mean = report["fields"]["zero"]["probes"][0]["statistics"]["mean"]
        self.assertIsNone(mean["target"])
        absolute = analyze_histories(np.arange(96), {"zero": history}, start_time=0, absolute_error=0.1)
        self.assertAlmostEqual(absolute["fields"]["zero"]["probes"][0]["statistics"]["mean"]["target"]["half_width_tolerance"], 0.1)
        values = np.sin(2 * np.pi * np.arange(128) / 16)
        periodic = history_from_groups([[value] for value in values])
        result = analyze_histories(np.arange(128), {"periodic": periodic}, start_time=0, max_lag=28)
        mean = result["fields"]["periodic"]["probes"][0]["statistics"]["mean"]
        self.assertIsNone(mean["observed_decorrelation_interval"])
        self.assertGreater(abs(mean["autocorrelation"][16]), 0.9)

    def test_report_uncertainty_targets_zero_means_and_json_cli(self):
        rng = np.random.default_rng(13)
        groups = [rng.normal(loc=2, size=2 + index % 3) for index in range(96)]
        history = history_from_groups(groups)
        times = 0.1 * np.arange(len(groups))
        report = analyze_histories(times, {"X.sigma0": history}, start_time=0)
        mean = report["fields"]["X.sigma0"]["probes"][0]["statistics"]["mean"]
        self.assertGreater(mean["standard_error"], 0)
        self.assertGreater(mean["effective_samples"], 0)
        self.assertLessEqual(mean["effective_samples"], len(times))
        self.assertAlmostEqual(mean["estimate"], np.concatenate(groups).mean())
        self.assertAlmostEqual(mean["confidence_interval"][1] - mean["estimate"], mean["confidence_half_width"])
        self.assertIn("estimated_required_snapshots_at_current_spacing", mean["target"])
        constant = history_from_groups([[3.0]] * len(times))
        constant_report = analyze_histories(times, {"constant": constant}, start_time=0)
        statistic = constant_report["fields"]["constant"]["probes"][0]["statistics"]["mean"]
        self.assertTrue(statistic["warnings"])
        self.assertNotIn("effective_samples", statistic)
        safe = json_safe(constant_report)
        json.dumps(safe, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "history.npz", Path(directory) / "report.json"
            arrays = {"diagnostics.times": times, **{f"diagnostics.X.sigma0.{name}": value for name, value in history.items()}}
            atomic_save(source, arrays)
            with patch("sys.argv", ["sampling", str(source), str(output), "--field", "X.sigma0", "--start-time", "0"]):
                main()
                result = json.loads(output.read_text())
                self.assertEqual(result["retained_snapshots"], len(times))
                self.assertEqual(set(result["fields"]), {"X.sigma0"})
                with self.assertRaises(FileExistsError):
                    main()


if __name__ == "__main__":
    unittest.main()
