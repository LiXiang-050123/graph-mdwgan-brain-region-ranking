from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "results" / "reference"
OURS = REFERENCE_ROOT / "corrected_secondary_ours"
BASELINES = REFERENCE_ROOT / "corrected_secondary_baselines"
OUTPUT = REPO_ROOT / "outputs" / "corrected_secondary_merged"

TABLES = (
    "perf_long.csv",
    "perf_mean_sd.csv",
    "stability_long.csv",
    "stability_mean_sd.csv",
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged_tables: dict[str, pd.DataFrame] = {}
    for filename in TABLES:
        frames = [pd.read_csv(folder / filename) for folder in (OURS, BASELINES)]
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(OUTPUT / filename, index=False)
        merged_tables[filename] = merged

    for folder in (OURS, BASELINES):
        for source in folder.glob("freq_top10_*.csv"):
            destination = OUTPUT / source.name
            if destination.exists():
                raise FileExistsError(f"Duplicate frequency output: {destination}")
            shutil.copy2(source, destination)

    perf = merged_tables["perf_long.csv"]
    stability = merged_tables["stability_long.csv"]
    expected_combinations = {
        (2, "Ours"),
        (4, "Ours"),
        (8, "Ours"),
        (8, "ROS"),
        (8, "SMOTE"),
        (8, "RealOnly"),
    }
    actual_combinations = set(zip(perf["factor"].astype(int), perf["method"].astype(str)))
    if actual_combinations != expected_combinations:
        raise AssertionError(
            f"Unexpected factor/method combinations: {sorted(actual_combinations)}"
        )
    if set(perf["fair"]) != {"raw"} or set(stability["fair"]) != {"raw"}:
        raise AssertionError("The corrected primary merge must contain raw results only")
    if perf["fold"].nunique() != 25 or stability["fold"].nunique() != 25:
        raise AssertionError("Expected exactly 25 outer folds")

    perf_key = ["fold", "factor", "method", "fair", "scheme", "k", "clf"]
    stability_key = ["fold", "factor", "method", "fair", "scheme", "k"]
    if perf.duplicated(perf_key).any():
        raise AssertionError("Duplicate rows found in perf_long.csv")
    if stability.duplicated(stability_key).any():
        raise AssertionError("Duplicate rows found in stability_long.csv")

    manifest = {
        "created": str(date.today()),
        "fix": "Five-method consensus (ANOVA, MI, LASSO, RF, RFE) with GNN-lite fused exactly once",
        "fair_modes": ["raw"],
        "outer_folds": 25,
        "bootstrap_per_fold": 50,
        "factor_method_combinations": sorted([list(item) for item in expected_combinations]),
        "perf_long_rows": int(len(perf)),
        "stability_long_rows": int(len(stability)),
        "source_directories": [str(OURS), str(BASELINES)],
    }
    (OUTPUT / "CORRECTED_RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
