"""Optional compact histories of the existing particle-ensemble accumulators."""

import numpy as np

ARRAYS = ("count", "mean", "m2", "m3", "m4")
PREFIX = "diagnostics."


def validate_diagnostics(config):
    """Normalize explicit probe selection without changing the main ensemble."""
    if "diagnostics" not in config:
        return
    options = config["diagnostics"]
    if not isinstance(options, dict):
        raise ValueError("diagnostics must be an object")
    unknown = set(options) - {"fields", "map_indices", "particle_loads"}
    if unknown:
        raise ValueError(f"Unknown diagnostics options: {sorted(unknown)}")
    fields = options.setdefault("fields", [])
    if not isinstance(fields, list) or any(not isinstance(name, str) or not name for name in fields) or len(fields) != len(set(fields)):
        raise ValueError("diagnostics.fields must contain distinct nonempty field names")
    selected = config.get("fields", [])
    if selected and set(fields) - set(selected):
        raise ValueError("diagnostics.fields must be included in the mapped fields")
    loads = options.setdefault("particle_loads", True)
    if type(loads) is not bool:
        raise ValueError("diagnostics.particle_loads must be a boolean")
    if not fields and not loads:
        raise ValueError("Select diagnostic fields or particle loads")
    indices = options.setdefault("map_indices", [])
    mapping = config["map"]
    if mapping["kind"] == "box":
        shape = (mapping["points"],) * 3
    elif mapping["kind"] == "axisymmetric":
        shape = (mapping["axial_points"], mapping["radial_points"])
    else:
        raise ValueError("Unknown diagnostic map kind")
    if not isinstance(indices, list) or (fields and not indices):
        raise ValueError("Provide diagnostics.map_indices for selected fields")
    for index in indices:
        if not isinstance(index, list) or len(index) != len(shape) or any(type(i) is not int or i < 0 or i >= size for i, size in zip(index, shape)):
            raise ValueError("diagnostics.map_indices must be valid zero-based map indices")
    if len({tuple(index) for index in indices}) != len(indices):
        raise ValueError("diagnostics.map_indices must be distinct")


def capture(moments, indices=None):
    """Copy before cumulative merging mutates the snapshot accumulator."""
    selection = tuple(np.asarray(indices, dtype=int).T) if indices is not None else ()
    return {name: np.atleast_1d(getattr(moments, name)[selection]).copy() for name in ARRAYS}


def history_arrays(records):
    """Pack in configured snapshot order after workers have been joined."""
    first = records[0]["diagnostics"]
    for record in records:
        if set(record["diagnostics"]) != set(first):
            raise ValueError("Inconsistent diagnostic fields between snapshots")
    times = np.asarray([record["time"] for record in records])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Diagnostics require finite, strictly increasing physical snapshot times")
    yield PREFIX + "times", times
    yield PREFIX + "steps", np.asarray([record["step"] for record in records], dtype=np.int64)
    for field in sorted(first):
        for name in ARRAYS:
            yield f"{PREFIX}{field}.{name}", np.stack([record["diagnostics"][field][name] for record in records])
