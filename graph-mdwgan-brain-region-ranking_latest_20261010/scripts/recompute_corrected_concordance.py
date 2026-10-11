from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "results" / "reference"
SOURCE = REFERENCE_ROOT / "pubmed_source_snapshot"
OUTPUT = REPO_ROOT / "outputs" / "pubmed_concordance_recomputed"
RANKING = REFERENCE_ROOT / "corrected_secondary" / "freq_top10_factor8_Ours_consensus_raw.csv"
PERMUTATIONS = 10_000
SEED = 20250829


def main() -> None:
    shutil.copytree(SOURCE, OUTPUT, dirs_exist_ok=True)

    summary = pd.read_csv(SOURCE / "literature_evidence_summary.csv")
    frequency = pd.read_csv(RANKING).rename(
        columns={
            "BrainRegion": "brain_region",
            "freq": "top10_frequency",
            "freq_rate": "stability",
        }
    )
    frequency["ml_rank"] = frequency["top10_frequency"].rank(
        ascending=False, method="min"
    ).astype(int)
    ml = frequency[["brain_region", "ml_rank", "top10_frequency", "stability"]]

    summary = summary.drop(
        columns=["ml_rank", "top10_frequency", "stability", "concordance_class"],
        errors="ignore",
    ).merge(ml, on="brain_region", how="left", validate="one_to_one")
    if summary["ml_rank"].isna().any():
        raise AssertionError("Corrected ML ranking does not cover all literature regions")

    merged = summary.dropna(subset=["ml_rank", "literature_composite_rank"]).copy()
    spearman = stats.spearmanr(merged["ml_rank"], merged["literature_composite_rank"])
    kendall = stats.kendalltau(merged["ml_rank"], merged["literature_composite_rank"])
    ml_top3 = set(merged.nsmallest(3, "ml_rank")["brain_region"])
    lit_top3 = set(merged.nsmallest(3, "literature_composite_rank")["brain_region"])
    ml_top5 = set(merged.nsmallest(5, "ml_rank")["brain_region"])
    lit_top5 = set(merged.nsmallest(5, "literature_composite_rank")["brain_region"])

    rng = np.random.default_rng(SEED)
    ml_rank = merged["ml_rank"].to_numpy(float)
    lit_rank = merged["literature_composite_rank"].to_numpy(float)
    more_extreme = sum(
        abs(stats.spearmanr(ml_rank, rng.permutation(lit_rank)).statistic)
        >= abs(spearman.statistic)
        for _ in range(PERMUTATIONS)
    )
    metrics = {
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "top3_overlap": len(ml_top3 & lit_top3) / 3,
        "top5_overlap": len(ml_top5 & lit_top5) / 5,
        "permutation_p": (more_extreme + 1) / (PERMUTATIONS + 1),
        "permutation_seed": SEED,
        "permutation_iterations": PERMUTATIONS,
    }

    ml_median = merged["ml_rank"].median()
    lit_median = merged["literature_composite_rank"].median()
    summary["concordance_class"] = "limited_evidence"
    summary.loc[
        (summary["ml_rank"] <= ml_median)
        & (summary["literature_composite_rank"] <= lit_median),
        "concordance_class",
    ] = "convergent"
    summary.loc[
        (summary["ml_rank"] <= ml_median)
        & (summary["literature_composite_rank"] > lit_median),
        "concordance_class",
    ] = "novel_candidate"
    summary.loc[
        (summary["ml_rank"] > ml_median)
        & (summary["literature_composite_rank"] <= lit_median),
        "concordance_class",
    ] = "literature_supported_lower_ml"

    summary.sort_values("ml_rank").to_csv(
        OUTPUT / "literature_evidence_summary.csv", index=False, encoding="utf-8-sig"
    )
    ml.sort_values("ml_rank").to_csv(
        OUTPUT / "ml_ranking_all_candidate_regions.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT / "concordance_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    provenance = {
        "literature_snapshot": str(SOURCE),
        "corrected_ml_ranking": str(RANKING),
        "note": "Literature records were held fixed; only the corrected ML ranking and concordance were recomputed.",
    }
    (OUTPUT / "CORRECTED_CONCORDANCE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
