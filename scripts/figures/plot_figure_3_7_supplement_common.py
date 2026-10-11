from __future__ import annotations

import gc
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_figure_3_7_literature_concordance as base
from plot_figure_3_7_ABCFG_clear import load_sources
from analysis_paths import CODE_ROOT, PUBMED_RESULTS


ROOT = CODE_ROOT
OUT = ROOT / "figure_3_7_supplement_common_300dpi_tiff"
OUT.mkdir(exist_ok=True)
CORRECTED_LITERATURE = PUBMED_RESULTS


TYPE_COLORS = {
    "Activity": base.COLORS["activity"],
    "Behavioral": base.COLORS["behavioral"],
    "Circuit": base.COLORS["direct"],
    "Intervention": base.COLORS["circuit"],
}


def set_clear_style() -> None:
    base.set_style()
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
        }
    )


def style_axes(ax: plt.Axes, left: bool = True, bottom: bool = True) -> None:
    base.style_axes(ax, left=left, bottom=bottom)


def add_panel_letter(fig: plt.Figure, label: str) -> None:
    fig.text(
        0.012,
        0.985,
        label,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="black",
    )


def norm01(series: pd.Series) -> pd.Series:
    arr = series.astype(float)
    span = arr.max() - arr.min()
    if span == 0:
        return arr * 0
    return (arr - arr.min()) / span


def load_pubmed_region_records() -> pd.DataFrame:
    csv_path = PUBMED_RESULTS / "pubmed_region_records_long.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    zip_path = ROOT / "pubmed_literature_concordance_package.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("pubmed_concordance_output/pubmed_region_records_long.csv") as fp:
            return pd.read_csv(fp)


def classify_evidence_type(value: object) -> str:
    text = str(value).lower()
    if "intervention" in text:
        return "Intervention"
    if "circuit" in text:
        return "Circuit"
    if "activity" in text:
        return "Activity"
    if "behavioral" in text:
        return "Behavioral"
    return "Other"


def panel_a_evidence_type_dotplot(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    df = evidence[evidence["brain_region"].astype(str).isin(base.CANDIDATES)].copy()
    df["brain_region"] = pd.Categorical(df["brain_region"].astype(str), base.CANDIDATES, ordered=True)
    df = df.sort_values("brain_region")
    ybase = np.arange(len(df))
    type_map = [
        ("Activity", "activity", -0.24),
        ("Behavioral", "behavioral", -0.08),
        ("Circuit", "circuit", 0.08),
        ("Intervention", "intervention", 0.24),
    ]
    for label, col, offset in type_map:
        y = ybase + offset
        x = df[col].astype(float).to_numpy()
        ax.hlines(y, 0, x, color=TYPE_COLORS[label], linewidth=2.0, alpha=0.48, zorder=1)
        ax.scatter(
            x,
            y,
            s=42,
            color=TYPE_COLORS[label],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    # Keep exact counts outside the main lollipop body to avoid covering marks.
    label_x0 = 208
    label_dx = 12.0
    label_cols = [
        ("A", "activity", TYPE_COLORS["Activity"], label_x0),
        ("B", "behavioral", TYPE_COLORS["Behavioral"], label_x0 + label_dx),
        ("C", "circuit", TYPE_COLORS["Circuit"], label_x0 + 2 * label_dx),
        ("I", "intervention", TYPE_COLORS["Intervention"], label_x0 + 3 * label_dx),
    ]
    for head, _, color, xpos in label_cols:
        ax.text(xpos, -0.58, head, ha="center", va="center",
                fontsize=7.2, fontweight="bold", color=color, clip_on=False)
    for yi, row in zip(ybase, df.itertuples()):
        for _, col, color, xpos in label_cols:
            ax.text(
                xpos,
                yi,
                f"{int(getattr(row, col))}",
                ha="center",
                va="center",
                fontsize=6.9,
                fontweight="bold",
                color=color,
                clip_on=False,
            )
    ax.set_yticks(ybase)
    ax.set_yticklabels(df["brain_region"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Number of studies")
    ax.set_xlim(0, 258)
    ax.grid(axis="x", color=base.COLORS["grid"], linewidth=0.45, alpha=0.70)
    handles = [
        Line2D([0], [0], marker="o", lw=1.8, color=TYPE_COLORS[label],
               markerfacecolor=TYPE_COLORS[label], markeredgecolor="white", label=label)
        for label, _, _ in type_map
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
        ncol=4,
        columnspacing=0.95,
        handlelength=1.1,
    )
    style_axes(ax)


def panel_b_pubmed_record_timeline(ax: plt.Axes, records: pd.DataFrame) -> None:
    df = records.copy()
    df = df[df["brain_region"].astype(str).isin(base.CANDIDATES)]
    df = df[(df["direct_meth_study_auto"] == True) & (df["original_study_auto"] == True)].copy()
    df["brain_region"] = pd.Categorical(df["brain_region"].astype(str), base.CANDIDATES, ordered=True)
    df["evidence_group"] = df["evidence_type_auto"].map(classify_evidence_type)
    df = df.sort_values(["brain_region", "pub_year", "pmid"])
    y_map = {region: idx for idx, region in enumerate(base.CANDIDATES)}
    rng = np.random.default_rng(3707)
    df["x_jitter"] = rng.normal(0, 0.10, len(df))
    df["y_jitter"] = rng.normal(0, 0.115, len(df))
    evidence_order = ["Activity", "Behavioral", "Circuit", "Intervention", "Other"]
    color_map = {
        "Activity": TYPE_COLORS["Activity"],
        "Behavioral": TYPE_COLORS["Behavioral"],
        "Circuit": TYPE_COLORS["Circuit"],
        "Intervention": TYPE_COLORS["Intervention"],
        "Other": "#AAB4BE",
    }
    for group in evidence_order:
        sub = df[df["evidence_group"] == group]
        if sub.empty:
            continue
        y = sub["brain_region"].astype(str).map(y_map).astype(float).to_numpy() + sub["y_jitter"].to_numpy()
        x = sub["pub_year"].astype(float).to_numpy() + sub["x_jitter"].to_numpy()
        ax.scatter(
            x,
            y,
            s=13 if group != "Other" else 10,
            color=color_map[group],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.58 if group != "Other" else 0.38,
            label=group,
            zorder=3,
        )
    counts = df["brain_region"].astype(str).value_counts()
    ax.set_yticks(np.arange(len(base.CANDIDATES)))
    ax.set_yticklabels([f"{region} ({int(counts.get(region, 0))})" for region in base.CANDIDATES])
    ax.invert_yaxis()
    ax.set_xlim(1979, 2027)
    ax.set_xticks([1980, 1990, 2000, 2010, 2020, 2026])
    ax.set_xlabel("Publication year")
    ax.set_ylabel("")
    ax.grid(axis="x", color=base.COLORS["grid"], linewidth=0.45, alpha=0.70)
    ax.grid(axis="y", color=base.COLORS["grid"], linewidth=0.35, alpha=0.45)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[group],
               markeredgecolor="white", markersize=5.0, label=group)
        for group in evidence_order
        if group in set(df["evidence_group"])
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        ncol=4,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    style_axes(ax)


def panel_c_candidate_metric_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    df = summary[summary["brain_region"].astype(str).isin(base.CANDIDATES)].copy()
    df["brain_region"] = pd.Categorical(df["brain_region"].astype(str), base.CANDIDATES, ordered=True)
    df = df.sort_values("brain_region")
    matrix = pd.DataFrame(
        {
            "Direct\nstudies": norm01(np.log10(df["direct_original_studies"].astype(float) + 1)).to_numpy(),
            "Intervention/\ncircuit": norm01(np.log10(df["intervention_or_circuit_studies_auto"].astype(float) + 1)).to_numpy(),
            "Activity": norm01(np.log10(df["activity_studies_auto"].astype(float) + 1)).to_numpy(),
            "Behavioral": norm01(np.log10(df["behavioral_studies_auto"].astype(float) + 1)).to_numpy(),
            "Specificity": norm01(df["specificity_ratio"].astype(float)).to_numpy(),
            "Co-occurrence": norm01(df["normalized_cooccurrence_strength"].astype(float)).to_numpy(),
            "ML stability": norm01(df["stability"].astype(float)).to_numpy(),
        },
        index=df["brain_region"].astype(str),
    )
    cmap = mpl.colormaps["YlGnBu"]
    xgrid, ygrid = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    values = matrix.to_numpy(dtype=float)
    sizes = 26 + 260 * values
    sc = ax.scatter(
        xgrid.ravel(),
        ygrid.ravel(),
        s=sizes.ravel(),
        c=values.ravel(),
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlim(-0.55, matrix.shape[1] - 0.45)
    ax.set_ylim(matrix.shape[0] - 0.45, -0.55)
    ax.grid(color=base.COLORS["grid"], linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.036, pad=0.025)
    cbar.set_label("Normalized value")
    cbar.ax.tick_params(labelsize=8.5, width=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color("#111111")


def panel_d_publication_window(ax: plt.Axes, summary: pd.DataFrame) -> None:
    df = summary[summary["brain_region"].astype(str).isin(base.CANDIDATES)].copy()
    df["brain_region"] = pd.Categorical(df["brain_region"].astype(str), base.CANDIDATES, ordered=True)
    df = df.sort_values("brain_region")
    y = np.arange(len(df))
    for yi, row in zip(y, df.itertuples()):
        region = str(row.brain_region)
        color = base.REGION_COLORS.get(region, base.COLORS["neutral"])
        first = float(row.first_pub_year)
        latest = float(row.latest_pub_year)
        mean = float(row.mean_pub_year)
        ax.plot(
            [first, latest],
            [yi, yi],
            color=color,
            linewidth=3.0,
            alpha=0.42,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            [first, latest],
            [yi, yi],
            s=34,
            color="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            mean,
            yi,
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            latest + 0.8,
            yi,
            f"{int(first)}-{int(latest)}",
            ha="left",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=color,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(df["brain_region"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Publication year")
    ax.set_xlim(1980, 2029)
    ax.set_xticks([1980, 1990, 2000, 2010, 2020, 2026])
    ax.grid(axis="x", color=base.COLORS["grid"], linewidth=0.45, alpha=0.70)
    handles = [
        Line2D([0], [0], color="#8BA8B8", lw=3, alpha=0.55, label="Publication window"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=base.COLORS["ml"],
               markeredgecolor="white", markersize=6, label="Mean publication year"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.2,
    )
    style_axes(ax)


def save_panel(label: str, stem: str, drawer, size: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=size)
    drawer(ax)
    fig.savefig(
        OUT / f"Fig3_7_supp_common_panel_{label}_{stem}_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        OUT / f"Fig3_7_supp_common_panel_{label}_{stem}_preview.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    fig.clf()
    plt.close(fig)
    gc.collect()

    fig, ax = plt.subplots(figsize=size)
    drawer(ax)
    add_panel_letter(fig, label)
    fig.savefig(
        OUT / f"Fig3_7_supp_common_panel_{label}_{stem}_lettered_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.clf()
    plt.close(fig)
    gc.collect()


def make_combined(summary: pd.DataFrame, evidence: pd.DataFrame, records: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.0, 10.2))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.03, 1.0],
        width_ratios=[1.02, 1.0],
        hspace=0.48,
        wspace=0.34,
    )
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
    }
    drawers = {
        "A": lambda ax: panel_a_evidence_type_dotplot(ax, evidence),
        "B": lambda ax: panel_b_pubmed_record_timeline(ax, records),
        "C": lambda ax: panel_c_candidate_metric_heatmap(ax, summary),
        "D": lambda ax: panel_d_publication_window(ax, summary),
    }
    for label, ax in axes.items():
        drawers[label](ax)
        ax.text(
            -0.075,
            1.08,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color="black",
            clip_on=False,
        )
    fig.savefig(
        OUT / "Fig3_7_supplement_common_combined_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        OUT / "Fig3_7_supplement_common_combined_preview.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.clf()
    plt.close(fig)
    gc.collect()


def main() -> None:
    set_clear_style()
    _, evidence = load_sources()
    summary = pd.read_csv(CORRECTED_LITERATURE / "literature_evidence_summary.csv")
    summary["brain_region"] = pd.Categorical(
        summary["brain_region"], base.REGION_ORDER, ordered=True
    )
    summary = summary.sort_values("brain_region").reset_index(drop=True)
    records = pd.read_csv(CORRECTED_LITERATURE / "pubmed_region_records_long.csv")
    summary.to_csv(OUT / "Fig3_7_supp_common_summary_source_data.csv", index=False)
    evidence.to_csv(OUT / "Fig3_7_supp_common_evidence_type_source_data.csv", index=False)
    records.to_csv(OUT / "Fig3_7_supp_common_pubmed_region_records_source_data.csv", index=False)
    panel_specs = [
        ("A", "evidence_type_dotplot", lambda ax: panel_a_evidence_type_dotplot(ax, evidence), (7.6, 4.0)),
        ("B", "pubmed_record_timeline", lambda ax: panel_b_pubmed_record_timeline(ax, records), (6.8, 3.4)),
        ("C", "candidate_metric_dotmatrix", lambda ax: panel_c_candidate_metric_heatmap(ax, summary), (6.0, 4.2)),
        ("D", "publication_window", lambda ax: panel_d_publication_window(ax, summary), (6.0, 4.2)),
    ]
    for label, stem, drawer, size in panel_specs:
        save_panel(label, stem, drawer, size)
    make_combined(summary, evidence, records)
    shutil.make_archive(str(ROOT / "figure_3_7_supplement_common_300dpi_tiff"), "zip", OUT)
    print(f"Saved common-style Figure 3.7 supplementary panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
