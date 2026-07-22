from __future__ import annotations

import math
import statistics


def summarize_timings(values: list[float]) -> dict[str, float]:
    """Return stable latency statistics in milliseconds."""
    if not values:
        raise ValueError("Cannot summarize an empty timing sequence.")
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def percentile(values: list[float], fraction: float) -> float:
    """Calculate an interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty sequence.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Percentile fraction must be between 0 and 1.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
