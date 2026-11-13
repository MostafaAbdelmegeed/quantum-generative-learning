"""Prespecified descriptive summaries for the ten-seed replication."""

import numpy as np


def paired_summary(differences):
    values = np.asarray(differences, dtype=float)
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise ValueError(
            "The replication decision requires ten finite paired differences"
        )
    improvements = int(np.count_nonzero(values < -1e-8))
    regressions = int(np.count_nonzero(values > 1e-8))
    mean = float(np.mean(values))
    return {
        "mean": mean,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "improvements": improvements,
        "regressions": regressions,
        "ties": 10 - improvements - regressions,
        "benefit_replicated": mean < -0.01 and improvements >= 8,
    }
