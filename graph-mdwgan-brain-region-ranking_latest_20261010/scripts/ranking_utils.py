"""Shared ranking logic used by the corrected secondary analyses."""

import numpy as np


BASE_RANKING_METHODS = ("ANOVA", "MI", "LASSO", "RF", "RFE")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based ascending ranks with average ranks for ties."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def consensus_and_fusion(
    base_scores: dict, gnn_score: np.ndarray, gnn_weight: float = 1.0 / 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build a five-method consensus and incorporate GNN-lite exactly once."""
    missing = set(BASE_RANKING_METHODS) - set(base_scores)
    extra = set(base_scores) - set(BASE_RANKING_METHODS)
    if missing or extra:
        raise ValueError(
            f"base_scores must contain exactly {BASE_RANKING_METHODS}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not 0.0 <= gnn_weight <= 1.0:
        raise ValueError("gnn_weight must be between 0 and 1")

    ranks = [
        _average_ranks(-np.asarray(base_scores[method], dtype=float))
        for method in BASE_RANKING_METHODS
    ]
    avg_rank = np.mean(np.vstack(ranks), axis=0)
    consensus_score = avg_rank.max() - avg_rank + 1.0

    gnn_score = np.asarray(gnn_score, dtype=float)
    if gnn_score.shape != consensus_score.shape:
        raise ValueError(
            f"gnn_score shape {gnn_score.shape} does not match "
            f"consensus shape {consensus_score.shape}"
        )

    def normalize(values: np.ndarray) -> np.ndarray:
        minimum, maximum = values.min(), values.max()
        return (values - minimum) / (maximum - minimum + 1e-12)

    fused_score = (
        (1.0 - gnn_weight) * normalize(consensus_score)
        + gnn_weight * normalize(gnn_score)
    )
    return consensus_score, fused_score
