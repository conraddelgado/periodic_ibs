"""Histogram probability densities of individual particle load realizations."""

import numpy as np

PDF_ARRAYS = ("bin_edges", "counts", "density", "sample_count", "samples")


def particle_pdfs(records, fields, bins):
    """Use global edges after all snapshots/workers have been combined.

    Only the small force/torque samples are retained, never per-snapshot maps.
    Keeping samples until this step avoids range clipping, approximate rebinning,
    averaging snapshot PDFs, or a second pass through the fluid restart files.
    """
    arrays, metadata = {}, {}
    samples = [np.asarray(record["particle_load_samples"], dtype=float) for record in records]
    if any(values.ndim != 2 or values.shape[1] != len(fields) for values in samples):
        raise ValueError("Invalid particle-load sample array for PDF construction")
    for column, field in enumerate(fields):
        pooled = np.concatenate([values[:, column] for values in samples]) if samples else np.empty(0)
        values = pooled[np.isfinite(pooled)]
        constant = bool(values.size and np.all(values == values[0]))
        if constant:
            # A constant component is a point mass, not a smooth density.
            # Supply a normalized one-bin representation and identify it in
            # metadata so plotting code can draw a vertical line at its value.
            value = float(values[0])
            padding = max(0.5, abs(value) * 1e-6)
            limit = np.finfo(float).max
            edges = np.array([max(-limit, value - padding), min(limit, value + padding)])
        elif values.size:
            # Very small fluctuations at a large offset can produce repeated
            # floating-point edges; never divide by a zero-width bin.
            edges = np.unique(np.histogram_bin_edges(values, bins=bins))
        else:
            edges = np.linspace(-0.5, 0.5, bins + 1)
        widths = np.diff(edges)
        if not len(widths) or np.any(widths <= 0) or not np.all(np.isfinite(widths)):
            raise ValueError(f"Cannot construct finite positive-width PDF bins for {field}")
        counts, edges = np.histogram(values, bins=edges)
        density = counts.astype(float) / values.size / widths if values.size else np.full(len(counts), np.nan)
        prefix = f"{field}.pdf."
        arrays[prefix + "bin_edges"] = edges
        arrays[prefix + "counts"] = counts.astype(np.int64)
        arrays[prefix + "density"] = density
        arrays[prefix + "sample_count"] = np.array(values.size, dtype=np.int64)
        arrays[prefix + "samples"] = values
        metadata[field] = {
            "finite_samples": int(values.size),
            "excluded_nonfinite_samples": int(pooled.size - values.size),
            "bin_count": len(counts),
            "point_mass": constant,
            "constant_value": float(values[0]) if constant else None,
        }
    return arrays, metadata
