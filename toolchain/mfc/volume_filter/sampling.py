"""Analyze compact closure histories; run with python -m mfc.volume_filter.sampling."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .diagnostics import ARRAYS, PREFIX
from .maps import Moments

STATISTICS = ("mean", "variance", "skewness", "kurtosis")
MIN_SAMPLES = 32


def regular_spacing(times):
    """Require physical time; never concatenate gaps or silently resample."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < MIN_SAMPLES or not np.all(np.isfinite(times)):
        raise ValueError(f"Provide at least {MIN_SAMPLES} finite physical snapshot times")
    differences = np.diff(times)
    spacing = float(np.median(differences))
    tolerance = 32 * np.finfo(float).eps * max(np.max(np.abs(times)), abs(spacing))
    if spacing <= 0 or np.any(differences <= 0) or not np.allclose(differences, spacing, rtol=1e-6, atol=tolerance):
        raise ValueError("Sampling analysis requires strictly increasing, uniformly spaced physical times")
    return spacing


def validate_history(history, snapshots):
    shape = np.asarray(history["count"]).shape
    if len(shape) != 2 or shape[0] != snapshots or shape[1] == 0:
        raise ValueError("Diagnostic moments must have shape (snapshot, probe)")
    for name in ARRAYS:
        value = np.asarray(history[name])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Inconsistent or nonfinite diagnostic {name}")
    count = history["count"]
    if np.any(count < 0) or np.any(count != np.floor(count)) or np.any(history["m2"] < 0) or np.any(history["m4"] < 0):
        raise ValueError("Invalid diagnostic counts or central sums")


def merge_rows(history, indices):
    result = Moments((history["count"].shape[1],))
    for index in indices:
        value = Moments(result.mean.shape)
        for name in ARRAYS:
            setattr(value, name, history[name][index])
        value.weight = value.count.astype(float)
        result.merge(value)
    return result


def statistics(moments):
    return {"mean": np.where(moments.weight > 0, moments.mean, np.nan), **{name: getattr(moments, name) for name in STATISTICS[1:]}}


def influence_series(history, total):
    """Delta-method influences of the pooled particle/time estimators.

    Count weighting treats the mean as a ratio of sums, including empty
    snapshots, rather than treating particle means as equally weighted samples.
    """
    count = history["count"].astype(float)
    denominator = np.where(count > 0, count, 1)
    delta = history["mean"] - total.mean
    second = (history["m2"] + count * delta**2) / denominator
    third = (history["m3"] + 3 * delta * history["m2"] + count * delta**3) / denominator
    fourth = (history["m4"] + 4 * delta * history["m3"] + 6 * delta**2 * history["m2"] + count * delta**4) / denominator
    variance, central3, central4 = (total.central_moment(order) for order in (2, 3, 4))
    safe_variance = np.where(variance > 0, variance, 1)
    variance_influence = second - variance
    third_influence = third - central3 - 3 * variance * delta
    fourth_influence = fourth - central4 - 4 * central3 * delta
    values = {
        "mean": delta,
        "variance": variance_influence,
        "skewness": third_influence / safe_variance**1.5 - 1.5 * central3 * variance_influence / safe_variance**2.5,
        "kurtosis": fourth_influence / safe_variance**2 - 2 * central4 * variance_influence / safe_variance**3,
    }
    average_count = count.mean(axis=0)
    weight = count / np.where(average_count > 0, average_count, 1)
    return {name: np.where(count > 0, weight * value, 0) for name, value in values.items()}


def autocorrelation(values, max_lag):
    """FFT autocorrelation with the N-k covariance normalization."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or max_lag < 1 or max_lag >= len(values):
        raise ValueError("Invalid autocorrelation input or maximum lag")
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if variance == 0:
        return np.full(max_lag + 1, np.nan)
    fft_size = 1 << (2 * len(values) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    covariance = np.fft.irfft(spectrum * spectrum.conj(), n=fft_size)[: max_lag + 1]
    return covariance / ((len(values) - np.arange(max_lag + 1)) * variance)


def correlation_factor(rho, snapshots):
    """Bartlett lag-window HAC estimate, including finite-record weighting."""
    lags = np.arange(1, len(rho))
    taper = 1 - lags / len(rho)
    return max(0.0, float(1 + 2 * np.sum(taper * (1 - lags / snapshots) * rho[1:])))


def mser_cutoff(values):
    """MSER-5 candidate; retain >=32 snapshots and discard at most half."""
    batch_size = 5
    batches = len(values) // batch_size
    limit = min(batches // 2, batches - 8, (len(values) - MIN_SAMPLES) // batch_size)
    if limit <= 0:
        return 0
    means = np.asarray(values[: batches * batch_size]).reshape(batches, batch_size).mean(axis=1)
    scores = [np.var(means[index:], ddof=1) / (batches - index) for index in range(limit + 1)]
    return int(np.argmin(scores)) * batch_size


def suggested_start(histories):
    """Screen counts and all estimators; this is not a stationarity proof."""
    candidates = {}
    for field, history in histories.items():
        total = merge_rows(history, range(len(history["count"])))
        series = {"count": history["count"], **influence_series(history, total)}
        for name, values in series.items():
            for probe in range(values.shape[1]):
                signal = values[:, probe]
                if np.all(np.isfinite(signal)):
                    candidates[f"{field}.probe{probe}.{name}"] = mser_cutoff(signal)
    return max(candidates.values(), default=0), candidates


def combine(left, right):
    result = Moments(left.mean.shape)
    result.merge(left)
    result.merge(right)
    return result


def block_jackknife(history, block_size):
    """Delete whole contiguous time blocks, preserving all particle scatter."""
    blocks = len(history["count"]) // block_size
    if blocks < 8:
        raise ValueError("Block analysis requires at least eight complete time blocks")
    groups = [merge_rows(history, range(index * block_size, (index + 1) * block_size)) for index in range(blocks)]
    empty = Moments(groups[0].mean.shape)
    prefix = [empty]
    for group in groups:
        prefix.append(combine(prefix[-1], group))
    suffix = [empty] * (blocks + 1)
    for index in range(blocks - 1, -1, -1):
        suffix[index] = combine(groups[index], suffix[index + 1])
    estimates = statistics(prefix[-1])
    replicates = [statistics(combine(prefix[index], suffix[index + 1])) for index in range(blocks)]
    errors = {}
    for name in STATISTICS:
        values = np.stack([replicate[name] for replicate in replicates])
        errors[name] = np.sqrt((blocks - 1) / blocks * np.sum((values - values.mean(axis=0)) ** 2, axis=0))
    return {"block_size": block_size, "blocks": blocks, "used_snapshots": blocks * block_size, "estimates": estimates, "standard_errors": errors}


def _block_sizes(snapshots):
    size = 1
    while size < max(1, snapshots // 64):
        size *= 2
    sizes = []
    while snapshots // size >= 8:
        sizes.append(size)
        size *= 2
    return sizes


def _window_statistics(history):
    return [{"start_index": int(window[0]), "stop_index": int(window[-1]) + 1, "statistics": statistics(merge_rows(history, window))} for window in np.array_split(np.arange(len(history["count"])), 4)]


def _analyze_signal(values, estimate, spacing, max_lag, confidence, relative_error, absolute_error, block_errors):
    result = {"estimate": float(estimate), "warnings": []}
    if not np.isfinite(estimate):
        result["warnings"].append("Statistic is undefined for the observed support or variance")
        return result
    rho = autocorrelation(values, max_lag)
    result["autocorrelation"] = rho
    if not np.all(np.isfinite(rho)):
        result["warnings"].append("Constant sampled influence: dependence and sampling accuracy are unresolved")
        return result
    snapshots = len(values)
    raw_factor = correlation_factor(rho, snapshots)
    factor = max(1.0, raw_factor)
    hac_error = np.std(values, ddof=1) * np.sqrt(factor / snapshots)
    finite_errors = [float(error) for error in block_errors if np.isfinite(error)]
    standard_error = max([float(hac_error), *finite_errors])
    uncertainty_factor = max(1.0, snapshots * standard_error**2 / np.var(values, ddof=1))
    z_score = norm.ppf((1 + confidence) / 2)
    half_width = float(z_score * standard_error)
    threshold = max(0.05, 2 / np.sqrt(snapshots))
    tail = np.abs(rho[max(1, len(rho) * 3 // 4) :])
    if np.max(tail) > threshold:
        result["warnings"].append("Correlation remains appreciable near the maximum lag; extend the record and lag window")
    shorter = correlation_factor(rho[: max(2, len(rho) // 2)], snapshots)
    if abs(raw_factor - shorter) > 0.25 * max(1.0, raw_factor, shorter):
        result["warnings"].append("Correlation time is sensitive to lag-window length")
    if finite_errors and max(finite_errors) > 1.5 * min(finite_errors):
        result["warnings"].append("Block uncertainty estimates have no clear plateau; extend the record")
    decorrelation = None
    # Require a substantial observed tail, not a zero crossing at the boundary.
    for lag in range(1, max_lag // 2 + 1):
        if np.max(np.abs(rho[lag:])) <= 0.05:
            decorrelation = float(lag * spacing)
            break
    numerically_zero = abs(estimate) <= 32 * np.finfo(float).eps * np.std(values, ddof=1)
    tolerance = absolute_error if absolute_error is not None else (0.0 if numerically_zero else relative_error * abs(estimate))
    if absolute_error is None and abs(estimate) < half_width:
        result["warnings"].append("Statistic is poorly separated from zero; consider a physically chosen absolute tolerance")
    target = None
    if tolerance > 0:
        required = int(np.ceil(snapshots * (half_width / tolerance) ** 2))
        target = {
            "half_width_tolerance": float(tolerance),
            "estimated_required_snapshots_at_current_spacing": max(2, required),
            "estimated_required_duration": float(max(1, required - 1) * spacing),
            "estimated_additional_duration": float(max(0, required - snapshots) * spacing),
            "met": bool(half_width <= tolerance),
        }
    else:
        result["warnings"].append("A zero mean or statistic needs an absolute error tolerance")
    result.update(
        correlation_factor=raw_factor,
        uncertainty_correlation_factor=float(uncertainty_factor),
        effective_samples=float(snapshots / uncertainty_factor),
        integral_time=0.5 * spacing * raw_factor,
        hac_standard_error=float(hac_error),
        standard_error=standard_error,
        confidence_interval=[float(estimate - half_width), float(estimate + half_width)],
        confidence_half_width=half_width,
        observed_decorrelation_interval=decorrelation,
        target=target,
    )
    return result


def analyze_histories(times, histories, start_time=None, max_lag=None, confidence=0.95, relative_error=0.05, absolute_error=None):
    """Return heuristic transient screens and correlation-aware error estimates.

    Forecasts assume stationary, mixing data and fixed physical sample spacing.
    They are provisional when lag-window or block-size checks remain sensitive.
    """
    spacing = regular_spacing(times)
    times = np.asarray(times, dtype=float)
    if not histories:
        raise ValueError("No diagnostic histories were selected")
    for history in histories.values():
        validate_history(history, len(times))
    if not 0 < confidence < 1 or not np.isfinite(relative_error) or relative_error <= 0:
        raise ValueError("Require 0 < confidence < 1 and positive relative_error")
    if absolute_error is not None and (not np.isfinite(absolute_error) or absolute_error <= 0):
        raise ValueError("absolute_error must be finite and positive")
    if start_time is not None and not np.isfinite(start_time):
        raise ValueError("start_time must be finite")
    suggested, candidates = suggested_start(histories)
    start = suggested if start_time is None else int(np.searchsorted(times, start_time))
    snapshots = len(times) - start
    if snapshots < MIN_SAMPLES:
        raise ValueError(f"Retain at least {MIN_SAMPLES} snapshots after the starting time")
    lag = max(2, min(snapshots // 4, 4096)) if max_lag is None else max_lag
    if type(lag) is not int or lag < 2 or lag > snapshots // 2:
        raise ValueError("max_lag must be an integer between 2 and half the retained snapshots")
    report = {
        "version": 1,
        "confidence": confidence,
        "spacing": spacing,
        "retained_snapshots": snapshots,
        "start_time": float(times[start]),
        "suggested_start_time": float(times[suggested]),
        "transient_candidates": {key: float(times[index]) for key, index in candidates.items()},
        "stationarity": "MSER-5 candidates and window statistics are diagnostics, not proof of stationarity",
        "correlation_method": "Bartlett lag-window HAC; uncertainty factor conservatively bounded below by one",
        "uncertainty_method": "Maximum of HAC and contiguous-block jackknife standard errors; asymptotic normal intervals",
        "max_lag": lag,
        "lag_times": np.arange(lag + 1) * spacing,
        "fields": {},
    }
    for field, full_history in histories.items():
        history = {name: value[start:] for name, value in full_history.items()}
        total = merge_rows(history, range(snapshots))
        estimates = statistics(total)
        influences = influence_series(history, total)
        blocks = [block_jackknife(history, size) for size in _block_sizes(snapshots)]
        probes = []
        for probe in range(history["count"].shape[1]):
            results = {}
            for name in STATISTICS:
                errors = [block["standard_errors"][name][probe] for block in blocks]
                results[name] = _analyze_signal(influences[name][:, probe], estimates[name][probe], spacing, lag, confidence, relative_error, absolute_error, errors)
            probes.append({"probe": probe, "valid_particle_time_pairs": int(total.count[probe]), "statistics": results})
        report["fields"][field] = {"probes": probes, "windows": _window_statistics(history), "block_analysis": blocks}
    return report


def load_histories(path, fields=None):
    with np.load(path, allow_pickle=False) as data:
        if PREFIX + "times" not in data:
            raise ValueError("Input has no diagnostics; enable diagnostics when processing the DNS snapshots")
        available = sorted(key[len(PREFIX) : -len(".count")] for key in data.files if key.startswith(PREFIX) and key.endswith(".count"))
        selected = available if fields is None else fields
        if set(selected) - set(available):
            raise ValueError(f"Unknown diagnostic fields: {sorted(set(selected) - set(available))}")
        histories = {field: {name: data[f"{PREFIX}{field}.{name}"].copy() for name in ARRAYS} for field in selected}
        return data[PREFIX + "times"].copy(), histories


def json_safe(value):
    """Write standard JSON, using null for unsupported/nonfinite estimates."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="New JSON report; existing files are never overwritten")
    parser.add_argument("--field", action="append", help="Exact diagnostic field, e.g. F_x.sigma0; repeat to select several")
    parser.add_argument("--start-time", type=float, help="Explicit physical production start; default uses the latest MSER-5 candidate")
    parser.add_argument("--max-lag", type=int)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--absolute-error", type=float, help="Override the relative tolerance for every selected statistic, in its own units")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    times, histories = load_histories(args.input, args.field)
    report = analyze_histories(times, histories, args.start_time, args.max_lag, args.confidence, args.relative_error, args.absolute_error)
    with args.output.open("x") as stream:
        json.dump(json_safe(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Sampling report saved: {args.output}")


if __name__ == "__main__":
    main()
