"""Multiple-testing adjustments: Holm (family-wise error) and Benjamini-Hochberg (FDR)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

Method = Literal["holm", "bh", "none"]


def adjust_pvalues(p_values: Sequence[float], method: Method = "holm") -> np.ndarray:
    """Adjusted p-values in the input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError("p-values must be a 1-d sequence in [0, 1]")
    m = p.size
    if m == 0 or method == "none":
        return p.copy()
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method == "bh":
        adj = np.minimum.accumulate(m / np.arange(m, 0, -1) * ranked[::-1])[::-1]
    else:
        raise ValueError(f"unknown method {method!r}")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out
