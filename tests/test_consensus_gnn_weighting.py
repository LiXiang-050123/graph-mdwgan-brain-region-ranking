import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ranking_utils import (
    BASE_RANKING_METHODS,
    consensus_and_fusion,
)


def main():
    base_scores = {
        method: np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        for method in BASE_RANKING_METHODS
    }
    gnn_forward = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    gnn_reverse = gnn_forward[::-1]

    consensus_a, fusion_a = consensus_and_fusion(
        base_scores, gnn_forward, gnn_weight=1.0 / 6.0
    )
    consensus_b, fusion_b = consensus_and_fusion(
        base_scores, gnn_reverse, gnn_weight=1.0 / 6.0
    )

    np.testing.assert_allclose(consensus_a, consensus_b)
    if np.allclose(fusion_a, fusion_b):
        raise AssertionError("Fusion should change when the GNN-lite score changes")

    contaminated = dict(base_scores)
    contaminated["GNN"] = gnn_forward
    try:
        consensus_and_fusion(contaminated, gnn_forward, gnn_weight=1.0 / 6.0)
    except ValueError:
        pass
    else:
        raise AssertionError("A GNN entry inside base_scores must be rejected")

    print("PASS: consensus uses five methods and GNN-lite is fused exactly once.")


if __name__ == "__main__":
    main()
