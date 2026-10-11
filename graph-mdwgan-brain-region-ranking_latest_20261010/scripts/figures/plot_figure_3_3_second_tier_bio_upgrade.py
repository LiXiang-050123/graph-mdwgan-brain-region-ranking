"""
Bioinformatics-style upgraded Figure 3.3 for the NAco/CLA second candidate tier.

The figure is rebuilt from the fold-level ranking files under cv_results.  The
panel set avoids relying on plain heatmaps and uses common manuscript plot
families: oncoprint-style selection matrix, raincloud distributions, UpSet
intersection summary, dot-matrix frequency plot, paired rank shifts, GNN
retention lollipops, and tier-gap forest summary.
"""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from analysis_paths import CODE_ROOT, PRIMARY_CV_RESULTS


ROOT = CODE_ROOT
CV = PRIMARY_CV_RESULTS
OUT = ROOT / "figure_3_3_second_tier_bio_upgrade_300dpi"
OUT.mkdir(exist_ok=True)

REGIONS = [
    "ACC", "AM", "BLA", "CeA", "CLA", "CM", "DMS", "GP", "IL",
    "LH", "MS", "NAco", "NAsh", "OFC", "PL", "RSC", "VP", "VTA",
]
CANDIDATES = ["ACC", "VTA", "NAco", "CLA", "BLA", "CeA", "DMS"]
SECOND_TIER = ["NAco", "CLA"]
RANK_METHODS = ["Consensus", "Fused", "GNN-lite"]
RANK_COL = {
    "Consensus": "consensus_rank",
    "Fused": "fused_rank",
    "GNN-lite": "gnn_rank",
}
SCORE_COLS = [
    ("ANOVA", "score_ANOVA"),
    ("LASSO", "score_LASSO"),
    ("RF", "score_RF"),
    ("RFE-SVM", "score_RFE"),
    ("MI", "score_MI"),
    ("Consensus", "ConsensusScore"),
    ("GNN-lite", "GNNlite_coef_norm"),
    ("Fused", "ConsensusPlusGNN"),
]
TIERS = {
    "ACC": "Tier 1", "VTA": "Tier 1",
    "NAco": "Tier 2", "CLA": "Tier 2",
    "BLA": "Supplementary", "CeA": "Supplementary", "DMS": "Supplementary",
}

COLORS = {
    "ink": "#123047",
    "axis": "#202428",
    "grid": "#E3E9EF",
    "panel": "#F7FAFB",
    "soft_blue": "#EDF5FA",
    "soft_green": "#E8F6F1",
    "soft_coral": "#FBEDEA",
    "naco": "#1FA88A",
    "cla": "#E06A4F",
    "tier1": "#265D78",
    "supp": "#C0913D",
    "blue": "#2E83B8",
    "gray": "#9AA3AD",
    "light_gray": "#D9E0E6",
    "dark_gray": "#59616A",
}
REGION_COLORS = {
    "ACC": COLORS["tier1"],
    "VTA": "#347C96",
    "NAco": COLORS["naco"],
    "CLA": COLORS["cla"],
    "BLA": COLORS["supp"],
    "CeA": "#D2A049",
    "DMS": "#B97935",
}
METHOD_COLORS = {
    "Consensus": COLORS["blue"],
    "Fused": COLORS["naco"],
    "GNN-lite": COLORS["supp"],
}
TIER_COLORS = {
    "Tier 1": COLORS["tier1"],
    "Tier 2": COLORS["naco"],
    "Supplementary": COLORS["supp"],
    "Other": COLORS["gray"],
}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.4,
        "axes.labelsize": 9.6,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.1,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.6,
        "ytick.major.size": 3.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
    })


def style_axes(ax: plt.Axes, top_right_off: bool = True) -> None:
    if top_right_off:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_color(COLORS["axis"])
    ax.tick_params(axis="both", colors="black", labelcolor="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontweight("bold")


def load_fold_rankings() -> pd.DataFrame:
    rows = []
    for fd in sorted(CV.glob("fold_*")):
        cons_path = fd / "fs_OURS" / "consensus_ranking.csv"
        fusion_path = fd / "fs_OURS" / "gnn_results" / "consensus_plus_gnn.csv"
        if not cons_path.exists() or not fusion_path.exists():
            continue
        fold = int(fd.name.split("_")[1])
        cons = pd.read_csv(cons_path)
        cons["fold"] = fold
        cons["consensus_rank"] = cons["avg_rank"].rank(method="first")
        fused = pd.read_csv(fusion_path).rename(columns={
            "ANOVA": "score_ANOVA",
            "LASSO": "score_LASSO",
            "RF": "score_RF",
            "RFE": "score_RFE",
            "MI": "score_MI",
        })
        fused["fold"] = fold
        fused["fused_rank"] = fused["ConsensusPlusGNN"].rank(ascending=False, method="first")
        fused["gnn_rank"] = fused["GNNlite_coef_norm"].rank(ascending=False, method="first")
        merged = cons[[
            "fold", "BrainRegion", "ANOVA", "LASSO", "RF", "RFE", "MI",
            "avg_rank", "consensus_rank",
        ]].merge(
            fused[[
                "fold", "BrainRegion", "score_ANOVA", "score_LASSO", "score_RF",
                "score_RFE", "score_MI", "ConsensusScore", "GNNlite_coef_norm",
                "ConsensusPlusGNN", "fused_rank", "gnn_rank",
            ]],
            on=["fold", "BrainRegion"],
        )
        rows.append(merged)
    if not rows:
        raise FileNotFoundError("No fold-level ranking files were found under cv_results.")
    df = pd.concat(rows, ignore_index=True)
    df["tier"] = df["BrainRegion"].map(TIERS).fillna("Other")
    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("BrainRegion").agg(
        mean_consensus_rank=("consensus_rank", "mean"),
        sd_consensus_rank=("consensus_rank", "std"),
        mean_fused_rank=("fused_rank", "mean"),
        sd_fused_rank=("fused_rank", "std"),
        mean_gnn_rank=("gnn_rank", "mean"),
        sd_gnn_rank=("gnn_rank", "std"),
        top3_consensus=("consensus_rank", lambda s: int((s <= 3).sum())),
        top10_consensus=("consensus_rank", lambda s: int((s <= 10).sum())),
        top3_fused=("fused_rank", lambda s: int((s <= 3).sum())),
        top10_fused=("fused_rank", lambda s: int((s <= 10).sum())),
        top3_gnn=("gnn_rank", lambda s: int((s <= 3).sum())),
        top10_gnn=("gnn_rank", lambda s: int((s <= 10).sum())),
    ).reset_index()
    summary["tier"] = summary["BrainRegion"].map(TIERS).fillna("Other")
    summary = summary.sort_values("mean_fused_rank")
    summary.to_csv(OUT / "figure_3_3_second_tier_bio_upgrade_summary.csv", index=False)
    return summary


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, 2.064 * sem


def half_violin(ax: plt.Axes, x: float, vals: np.ndarray, color: str, side: str) -> None:
    parts = ax.violinplot(vals, positions=[x], widths=0.58, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(1.0)
        verts = body.get_paths()[0].vertices
        if side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], x)
        else:
            verts[:, 0] = np.maximum(verts[:, 0], x)


def draw_panel_a_oncoprint(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    events = [
        ("NAco", "Consensus Top-3", "consensus_rank", 3, METHOD_COLORS["Consensus"]),
        ("NAco", "Fused Top-3", "fused_rank", 3, METHOD_COLORS["Fused"]),
        ("NAco", "GNN-lite Top-10", "gnn_rank", 10, METHOD_COLORS["GNN-lite"]),
        ("CLA", "Consensus Top-3", "consensus_rank", 3, METHOD_COLORS["Consensus"]),
        ("CLA", "Fused Top-3", "fused_rank", 3, METHOD_COLORS["Fused"]),
        ("CLA", "GNN-lite Top-10", "gnn_rank", 10, METHOD_COLORS["GNN-lite"]),
    ]
    folds = np.arange(1, 26)
    ax.set_xlim(-5.2, 27.9)
    ax.set_ylim(-0.5, len(events) - 0.5)
    for i, (region, label, col, k, color) in enumerate(events):
        y = len(events) - 1 - i
        selected = (
            df[df["BrainRegion"].eq(region)]
            .sort_values("fold")[col]
            .to_numpy(dtype=float) <= k
        )
        ax.add_patch(Rectangle((0.5, y - 0.36), 25, 0.72,
                               facecolor="#F4F7F8", edgecolor="none", zorder=0))
        for fold, is_selected in zip(folds, selected):
            face = color if is_selected else "white"
            edge = color if is_selected else COLORS["light_gray"]
            alpha = 0.95 if is_selected else 1.0
            ax.add_patch(Rectangle((fold - 0.38, y - 0.28), 0.76, 0.56,
                                   facecolor=face, edgecolor=edge, linewidth=0.55,
                                   alpha=alpha, zorder=2))
        ax.text(-4.65, y, region, ha="right", va="center",
                fontsize=8.8, fontweight="bold", color=REGION_COLORS[region])
        ax.text(-4.05, y, label, ha="left", va="center",
                fontsize=7.2, fontweight="bold", color=COLORS["ink"])
        ax.text(26.1, y, f"{selected.sum()}/25", ha="left", va="center",
                fontsize=8.0, fontweight="bold", color=color)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("Cross-validation split")
    ax.set_yticks([])
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=METHOD_COLORS["Consensus"],
               markeredgecolor="none", markersize=6.8, label="Consensus"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=METHOD_COLORS["Fused"],
               markeredgecolor="none", markersize=6.8, label="Fused"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=METHOD_COLORS["GNN-lite"],
               markeredgecolor="none", markersize=6.8, label="GNN-lite"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.52, 1.025), ncol=3, columnspacing=0.9,
              handletextpad=0.35)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def draw_panel_b_raincloud(ax: plt.Axes, df: pd.DataFrame) -> None:
    positions = {"Consensus": 0, "Fused": 1, "GNN-lite": 2}
    rng = np.random.default_rng(3402)
    ax.axhspan(0.5, 3.5, color=COLORS["soft_green"], alpha=0.62, zorder=0)
    ax.axhspan(3.5, 10.5, color=COLORS["soft_blue"], alpha=0.38, zorder=0)
    for method, x in positions.items():
        for region, side, dx, marker in [("NAco", "left", -0.12, "o"), ("CLA", "right", 0.12, "D")]:
            vals = df[df["BrainRegion"].eq(region)][RANK_COL[method]].to_numpy(dtype=float)
            color = REGION_COLORS[region]
            half_violin(ax, x + dx, vals, color, side)
            jitter = rng.normal(0, 0.030, len(vals))
            ax.scatter(np.full(len(vals), x + dx) + jitter, vals, s=13,
                       color=color, alpha=0.46, edgecolor="white",
                       linewidth=0.25, zorder=3)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.plot([x + dx - 0.16, x + dx + 0.16], [med, med],
                    color=color, linewidth=1.65, zorder=4)
            ax.plot([x + dx, x + dx], [q1, q3],
                    color=color, linewidth=3.0, alpha=0.45, zorder=4)
            mean = vals.mean()
            ax.scatter([x + dx], [mean], s=42, marker=marker,
                       color=color, edgecolor="white", linewidth=0.7, zorder=5)
            ax.text(x + dx, mean - 0.62, f"{mean:.2f}",
                    ha="center", va="center", fontsize=7.0,
                    fontweight="bold", color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.22, "alpha": 0.82})
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(10.8, 0.45)
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(list(positions.keys()))
    ax.set_yticks([1, 3, 5, 7, 10])
    ax.set_ylabel("Rank")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.82)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["naco"],
               markeredgecolor="white", markersize=6.2, label="NAco"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["cla"],
               markeredgecolor="white", markersize=5.8, label="CLA"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.66, 1.025), ncol=2, columnspacing=0.9,
              handletextpad=0.35)
    style_axes(ax)


def draw_panel_c_upset(ax: plt.Axes, df: pd.DataFrame) -> None:
    records = []
    for region in SECOND_TIER:
        sub = df[df["BrainRegion"].eq(region)].sort_values("fold")
        for _, row in sub.iterrows():
            records.append({
                "Region": region,
                "C Top-3": bool(row["consensus_rank"] <= 3),
                "F Top-3": bool(row["fused_rank"] <= 3),
                "G Top-10": bool(row["gnn_rank"] <= 10),
            })
    event_df = pd.DataFrame(records)
    cats = ["C Top-3", "F Top-3", "G Top-10"]
    combo_rows = []
    for region in SECOND_TIER:
        sub = event_df[event_df["Region"].eq(region)]
        grouped = sub.groupby(cats).size().reset_index(name="count")
        for _, row in grouped.iterrows():
            if row["count"] > 0:
                combo_rows.append({
                    "Region": region,
                    "pattern": tuple(bool(row[c]) for c in cats),
                    "count": int(row["count"]),
                })
    combos = pd.DataFrame(combo_rows).sort_values(["Region", "count"], ascending=[True, False])
    x = np.arange(len(combos))
    ax.bar(x, combos["count"], color=[REGION_COLORS[r] for r in combos["Region"]],
           width=0.62, alpha=0.82, edgecolor="white", linewidth=0.8)
    for xi, (_, row) in zip(x, combos.iterrows()):
        ax.text(xi, row["count"] + 0.35, str(int(row["count"])),
                ha="center", va="bottom", fontsize=7.3,
                fontweight="bold", color=REGION_COLORS[row["Region"]])
    ax.set_ylabel("Splits")
    ax.set_ylim(-4.2, max(8, float(combos["count"].max()) + 3.0))
    ax.set_xticks([])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.80)
    ydot = [-1.0, -2.0, -3.0]
    for xi, (_, row) in zip(x, combos.iterrows()):
        pattern = row["pattern"]
        color = REGION_COLORS[row["Region"]]
        active_y = []
        for yi, active in zip(ydot, pattern):
            ax.scatter([xi], [yi], s=34 if active else 22,
                       facecolor=color if active else "white",
                       edgecolor=color if active else COLORS["light_gray"],
                       linewidth=0.8, zorder=3)
            if active:
                active_y.append(yi)
        if len(active_y) >= 2:
            ax.plot([xi, xi], [min(active_y), max(active_y)], color=color,
                    linewidth=1.25, alpha=0.80, zorder=2)
        ax.text(xi, -3.72, row["Region"], ha="center", va="top",
                fontsize=6.9, fontweight="bold", color=color)
    for yi, cat in zip(ydot, cats):
        ax.text(-0.62, yi, cat, ha="right", va="center",
                fontsize=7.0, fontweight="bold", color=COLORS["ink"])
    style_axes(ax)
    ax.spines["bottom"].set_visible(False)


def draw_panel_d_frequency_dotplot(ax: plt.Axes, summary: pd.DataFrame) -> None:
    metrics = [
        ("Consensus Top-3", "top3_consensus", METHOD_COLORS["Consensus"]),
        ("Consensus Top-10", "top10_consensus", METHOD_COLORS["Consensus"]),
        ("Fused Top-3", "top3_fused", METHOD_COLORS["Fused"]),
        ("Fused Top-10", "top10_fused", METHOD_COLORS["Fused"]),
        ("GNN-lite Top-3", "top3_gnn", METHOD_COLORS["GNN-lite"]),
        ("GNN-lite Top-10", "top10_gnn", METHOD_COLORS["GNN-lite"]),
    ]
    row = summary.set_index("BrainRegion")
    ymap = {"NAco": 1, "CLA": 0}
    for i, (label, col, color) in enumerate(metrics):
        for region in SECOND_TIER:
            count = int(row.loc[region, col])
            freq = 100 * count / 25
            size = 52 + 280 * freq / 100
            ax.scatter(i, ymap[region], s=size, color=color, alpha=0.82,
                       edgecolor="white", linewidth=0.85, zorder=3)
            txt_color = "white" if freq >= 68 else COLORS["ink"]
            ax.text(i, ymap[region], f"{count}/25", ha="center", va="center",
                    fontsize=6.9, fontweight="bold", color=txt_color, zorder=4)
    ax.set_xlim(-0.65, len(metrics) - 0.35)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["NAco", "CLA"])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[0].replace(" ", "\n") for m in metrics], rotation=0)
    ax.tick_params(axis="x", labelsize=6.8, pad=2)
    for y in [0, 1]:
        ax.axhline(y, color=COLORS["grid"], linewidth=0.6, zorder=0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    style_axes(ax)


def draw_panel_e_rank_shift(ax: plt.Axes, summary: pd.DataFrame) -> None:
    sub = summary[summary["BrainRegion"].isin(CANDIDATES)].copy()
    sub = sub.sort_values("mean_fused_rank", ascending=False)
    y = np.arange(len(sub))
    for yi, row in enumerate(sub.itertuples(index=False)):
        color = REGION_COLORS.get(row.BrainRegion, COLORS["gray"])
        ax.plot([row.mean_consensus_rank, row.mean_fused_rank], [yi, yi],
                color=COLORS["light_gray"], linewidth=2.1, zorder=1)
        ax.scatter([row.mean_consensus_rank], [yi], s=48, facecolor="white",
                   edgecolor=color, linewidth=1.1, marker="o", zorder=3)
        ax.scatter([row.mean_fused_rank], [yi], s=58, color=color,
                   edgecolor="white", linewidth=0.8, marker="D", zorder=4)
        if row.BrainRegion in SECOND_TIER:
            ax.text(row.mean_fused_rank + 0.38, yi, f"{row.mean_fused_rank:.2f}",
                    ha="left", va="center", fontsize=7.5,
                    fontweight="bold", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["BrainRegion"].tolist())
    for tick, region in zip(ax.get_yticklabels(), sub["BrainRegion"].tolist()):
        tick.set_color(REGION_COLORS.get(region, COLORS["dark_gray"]))
        tick.set_fontweight("bold")
    ax.set_xlim(1.0, 9.2)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xlabel("Mean rank")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.80)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["dark_gray"], markersize=5.8, label="Consensus"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["dark_gray"],
               markeredgecolor="white", markersize=5.8, label="Fused"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.60, 1.025), ncol=2,
              columnspacing=0.9, handletextpad=0.35)
    style_axes(ax)


def draw_panel_f_gnn_lollipop(ax: plt.Axes, summary: pd.DataFrame) -> None:
    row = summary.set_index("BrainRegion")
    x = np.arange(len(SECOND_TIER))
    vals = np.array([100 * row.loc[r, "top10_gnn"] / 25 for r in SECOND_TIER], dtype=float)
    counts = [int(row.loc[r, "top10_gnn"]) for r in SECOND_TIER]
    ax.axhline(80, color=COLORS["dark_gray"], linewidth=0.85,
               linestyle="--", alpha=0.65, zorder=1)
    ax.text(1.42, 80, "80%", ha="right", va="bottom",
            fontsize=7.3, fontweight="bold", color=COLORS["dark_gray"])
    for xi, region, val, count in zip(x, SECOND_TIER, vals, counts):
        color = REGION_COLORS[region]
        ax.vlines(xi, 0, val, color=color, linewidth=5.2, alpha=0.20, zorder=2)
        ax.scatter([xi], [val], s=148, color=color, edgecolor="white",
                   linewidth=1.1, zorder=3)
        ax.text(xi, val + 5.0, f"{count}/25", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(SECOND_TIER)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("GNN-lite Top-10 frequency (%)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.82)
    style_axes(ax)


def draw_panel_g_tier_gap_forest(ax: plt.Axes, df: pd.DataFrame) -> None:
    piv = df.pivot(index="fold", columns="BrainRegion", values="fused_rank").sort_index()
    tier1 = piv[["ACC", "VTA"]].mean(axis=1)
    tier2 = piv[["NAco", "CLA"]].mean(axis=1)
    supp = piv[["BLA", "CeA", "DMS"]].mean(axis=1)
    gaps = pd.DataFrame({
        "Tier 2 - Tier 1": tier2 - tier1,
        "Supplementary - Tier 2": supp - tier2,
    })
    y = np.arange(len(gaps.columns))[::-1]
    colors = [COLORS["naco"], COLORS["supp"]]
    ax.axvline(0, color=COLORS["axis"], linewidth=0.9, alpha=0.75)
    rng = np.random.default_rng(202608)
    for yi, (col, color) in zip(y, zip(gaps.columns, colors)):
        vals = gaps[col].to_numpy(dtype=float)
        ax.scatter(vals, np.full(len(vals), yi) + rng.normal(0, 0.045, len(vals)),
                   s=18, color=color, alpha=0.48, edgecolor="white",
                   linewidth=0.35, zorder=2)
        mean, ci = ci95(vals)
        ax.errorbar(mean, yi, xerr=ci, fmt="o", markersize=7.2,
                    color=color, ecolor=color, capsize=5.0,
                    markeredgecolor="white", markeredgewidth=0.85,
                    elinewidth=1.4, zorder=4)
        ax.text(mean + ci + 0.16, yi, f"{mean:.2f}", ha="left", va="center",
                fontsize=8.0, fontweight="bold", color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.22, "alpha": 0.88})
    ax.set_yticks(y)
    ax.set_yticklabels(list(gaps.columns))
    ax.set_xlabel("Mean fused-rank gap with 95% CI")
    ax.set_xlim(-1.0, 5.4)
    ax.set_xticks([-1, 0, 1, 2, 3, 4, 5])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.82)
    style_axes(ax)


def draw_panel_h_priority_fingerprint(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = [name for name, _ in SCORE_COLS]
    cols = [col for _, col in SCORE_COLS]
    all_scores = df[cols].to_numpy(dtype=float)
    vmin = np.nanmin(all_scores, axis=0)
    vmax = np.nanmax(all_scores, axis=0)
    ymap = {"NAco": 1, "CLA": 0}
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(-0.6, 1.6)
    ax.axvspan(4.5, 7.5, color=COLORS["soft_blue"], alpha=0.55, zorder=0)
    for region in SECOND_TIER:
        means = df[df["BrainRegion"].eq(region)][cols].mean().to_numpy(dtype=float)
        scaled = np.clip((means - vmin) / np.where(vmax - vmin == 0, 1, vmax - vmin), 0, 1)
        color = REGION_COLORS[region]
        for i, val in enumerate(scaled):
            size = 62 + 240 * float(val)
            ax.scatter(i, ymap[region], s=size, color=color, alpha=0.82,
                       edgecolor="white", linewidth=0.85, zorder=3)
            ax.text(i, ymap[region], f"{val:.2f}", ha="center", va="center",
                    fontsize=6.45, fontweight="bold",
                    color="white" if val >= 0.62 else COLORS["ink"], zorder=4)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["NAco", "CLA"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylabel("")
    ax.set_xlabel("Direction-aligned normalized priority score")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    for y in [0, 1]:
        ax.axhline(y, color=COLORS["grid"], linewidth=0.55, zorder=0)
    style_axes(ax)


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT / f"{stem}_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def make_individual(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    panels = [
        ("panel_A_oncoprint_selection_matrix", draw_panel_a_oncoprint, (6.3, 3.2), (df, summary)),
        ("panel_B_rank_raincloud", draw_panel_b_raincloud, (4.8, 3.25), (df,)),
        ("panel_C_upset_intersections", draw_panel_c_upset, (4.6, 3.35), (df,)),
        ("panel_D_topk_frequency_dotplot", draw_panel_d_frequency_dotplot, (6.2, 3.0), (summary,)),
        ("panel_E_consensus_fused_rank_shift", draw_panel_e_rank_shift, (4.8, 3.4), (summary,)),
        ("panel_F_gnn_lollipop", draw_panel_f_gnn_lollipop, (3.7, 3.15), (summary,)),
        ("panel_G_tier_gap_forest", draw_panel_g_tier_gap_forest, (4.6, 2.85), (df,)),
        ("panel_H_priority_fingerprint_dotplot", draw_panel_h_priority_fingerprint, (6.0, 2.65), (df,)),
    ]
    for stem, drawer, size, args in panels:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax, *args)
        save_all(fig, stem)


def make_combined(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15.8, 9.8))
    gs = GridSpec(
        3, 24, figure=fig,
        height_ratios=[1.05, 1.08, 0.92],
        width_ratios=[1] * 24,
        wspace=1.20,
        hspace=0.46,
    )
    axes = [
        fig.add_subplot(gs[0, 0:10]),
        fig.add_subplot(gs[0, 10:17]),
        fig.add_subplot(gs[0, 17:24]),
        fig.add_subplot(gs[1, 0:10]),
        fig.add_subplot(gs[1, 10:17]),
        fig.add_subplot(gs[1, 17:24]),
        fig.add_subplot(gs[2, 0:8]),
        fig.add_subplot(gs[2, 8:24]),
    ]
    draw_panel_a_oncoprint(axes[0], df, summary)
    draw_panel_b_raincloud(axes[1], df)
    draw_panel_c_upset(axes[2], df)
    draw_panel_d_frequency_dotplot(axes[3], summary)
    draw_panel_e_rank_shift(axes[4], summary)
    draw_panel_f_gnn_lollipop(axes[5], summary)
    draw_panel_g_tier_gap_forest(axes[6], df)
    draw_panel_h_priority_fingerprint(axes[7], df)
    fig.savefig(OUT / "figure_3_3_second_tier_bio_upgrade_combined.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure_3_3_second_tier_bio_upgrade_combined.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure_3_3_second_tier_bio_upgrade_combined.svg",
                bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT / "figure_3_3_second_tier_bio_upgrade_combined_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def zip_outputs() -> Path:
    zip_path = ROOT / "figure_3_3_second_tier_bio_upgrade_300dpi_tiff.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for path in sorted(OUT.glob("*")):
            if path.suffix.lower() in {".tif", ".tiff", ".png", ".pdf", ".svg", ".csv"}:
                zf.write(path, arcname=f"{OUT.name}/{path.name}")
    return zip_path


def main() -> None:
    set_style()
    df = load_fold_rankings()
    summary = make_summary(df)
    make_individual(df, summary)
    make_combined(df, summary)
    zip_path = zip_outputs()
    print(f"Saved upgraded Figure 3.3 panels to {OUT.resolve()}")
    print(f"Packaged outputs: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
