"""Reconstruct the 24 x 18 matrix and retain behavioral-paradigm labels."""

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data" / "source" / "Fig1.xlsx"
LONG_OUTPUT = REPO_ROOT / "data" / "brain_region_cfos_with_paradigm.csv"
LEGACY_OUTPUT = REPO_ROOT / "data" / "brain_region_cfos_reconstructed.xlsx"
EXPECTED_LEGACY = REPO_ROOT / "data" / "brain_region_cfos.xlsx"


def extract_sheet(sheet: str, paradigm: str) -> pd.DataFrame:
    raw = pd.read_excel(SOURCE, sheet_name=sheet)
    regions = raw.iloc[:, 0].astype(str)
    meth_start = next(
        index for index, column in enumerate(raw.columns) if str(column).startswith("METH")
    )

    frames = []
    for group, start, stop in (
        ("Saline", 1, meth_start),
        ("METH", meth_start, raw.shape[1]),
    ):
        values = raw.iloc[:, start:stop].dropna(axis=1, how="all")
        values.index = regions
        animals = values.T.reset_index(drop=True)
        animals.columns = regions
        animals.insert(0, "Paradigm", paradigm)
        animals.insert(0, "Group", group)
        frames.append(animals)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    self_administration = extract_sheet("Fig1d", "Self-administration")
    reinstatement = extract_sheet("Fig1e", "Cue-induced reinstatement")
    # Preserve the historical pooled-matrix column order: Saline SA, Saline CS,
    # METH CS, then METH SA. The source sheets themselves use Saline then METH.
    combined = pd.concat(
        [
            self_administration[self_administration["Group"].eq("Saline")],
            reinstatement[reinstatement["Group"].eq("Saline")],
            reinstatement[reinstatement["Group"].eq("METH")],
            self_administration[self_administration["Group"].eq("METH")],
        ],
        ignore_index=True,
    )
    combined["Sample ID"] = ""
    for group in ("Saline", "METH"):
        mask = combined["Group"].eq(group)
        combined.loc[mask, "Sample ID"] = [
            f"{group}_{index:02d}" for index in range(1, int(mask.sum()) + 1)
        ]
    combined = combined[["Sample ID", "Group", "Paradigm", *combined.columns[2:-1]]]

    expected_counts = {
        ("Saline", "Self-administration"): 6,
        ("METH", "Self-administration"): 7,
        ("Saline", "Cue-induced reinstatement"): 5,
        ("METH", "Cue-induced reinstatement"): 6,
    }
    counts = combined.groupby(["Group", "Paradigm"]).size().to_dict()
    if counts != expected_counts or combined.shape != (24, 21):
        raise AssertionError(f"Unexpected reconstructed matrix: shape={combined.shape}, counts={counts}")

    combined.to_csv(LONG_OUTPUT, index=False, encoding="utf-8-sig")

    regions = list(combined.columns[3:])
    ordered = pd.concat(
        [combined[combined["Group"].eq("Saline")], combined[combined["Group"].eq("METH")]],
        ignore_index=True,
    )
    legacy = ordered.set_index("Sample ID")[regions].T
    legacy.columns = ordered["Group"].tolist()
    legacy.to_excel(LEGACY_OUTPUT)

    expected = pd.read_excel(EXPECTED_LEGACY, index_col=0)
    if expected.shape != legacy.shape or not np.allclose(
        expected.to_numpy(float), legacy.to_numpy(float), rtol=0, atol=1e-10
    ):
        raise AssertionError("Reconstruction does not match data/brain_region_cfos.xlsx")
    print(f"Wrote {LONG_OUTPUT.relative_to(REPO_ROOT)} and verified the legacy matrix.")


if __name__ == "__main__":
    main()
