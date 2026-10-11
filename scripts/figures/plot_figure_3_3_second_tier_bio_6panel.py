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
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath

from analysis_paths import CODE_ROOT, PRIMARY_CV_RESULTS


ROOT = CODE_ROOT
CV = PRIMARY_CV_RESULTS
OUT = ROOT / "figure_3_3_second_tier_bio_6panel_300dpi"
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
    ax.set_xlim(-9.7, 29.1)
    ax.set_ylim(-0.5, len(events) - 0.5)
    for y0, color in [(2.5, COLORS["naco"]), (-0.5, COLORS["cla"])]:
        ax.add_patch(Rectangle(
            (-9.05, y0), 35.30, 3.0,
            facecolor=color, edgecolor="none", alpha=0.035, zorder=-2,
        ))
    group_centers = {"NAco": 4.0, "CLA": 1.0}
    for region, yc in group_centers.items():
        color = REGION_COLORS[region]
        ax.add_patch(FancyBboxPatch(
            (-8.92, yc - 1.18), 0.09, 2.36,
            boxstyle="round,pad=0.003,rounding_size=0.045",
            facecolor=color, edgecolor="none", alpha=0.88, zorder=1,
        ))
        ax.text(-8.55, yc, region, ha="left", va="center",
                fontsize=9.2, fontweight="bold", color=color)
    ax.text(26.95, 5.62, "Retained\nsplits", ha="center", va="bottom",
            fontsize=6.3, fontweight="bold", color=COLORS["dark_gray"],
            linespacing=0.86)
    for i, (region, label, col, k, color) in enumerate(events):
        y = len(events) - 1 - i
        method_label = label
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
            ax.add_patch(FancyBboxPatch(
                (fold - 0.37, y - 0.27), 0.74, 0.54,
                boxstyle="round,pad=0.015,rounding_size=0.035",
                facecolor=face, edgecolor=edge, linewidth=0.58,
                alpha=alpha, zorder=2,
            ))
        ax.text(-0.30, y, method_label, ha="right", va="center",
                fontsize=6.95, fontweight="bold", color=COLORS["ink"])
        count = int(selected.sum())
        ax.text(27.55, y, f"{count}/25", ha="left", va="center",
                fontsize=7.7, fontweight="bold", color=color)
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
              bbox_to_anchor=(0.54, 1.045), ncol=3, columnspacing=0.9,
              handletextpad=0.35)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def draw_panel_b_raincloud(ax: plt.Axes, df: pd.DataFrame) -> None:
    methods = ["Consensus", "Fused", "GNN-lite"]
    y_positions = {"Consensus": 2, "Fused": 1, "GNN-lite": 0}
    panel_colors = {"NAco": COLORS["blue"], "CLA": COLORS["naco"]}
    rng = np.random.default_rng(3402)
    for method in methods:
        y0 = y_positions[method]
        ax.hlines(y0, 0.75, 10.25, color=COLORS["grid"], linewidth=0.75, zorder=1)
        for region, dy, marker in [("NAco", 0.15, "o"), ("CLA", -0.15, "D")]:
            vals = df[df["BrainRegion"].eq(region)][RANK_COL[method]].to_numpy(dtype=float)
            color = panel_colors[region]
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            vmin, vmax = np.min(vals), np.max(vals)
            mean = vals.mean()
            yy = y0 + dy
            ax.plot([vmin, vmax], [yy, yy], color=color, linewidth=1.0,
                    alpha=0.54, solid_capstyle="round", zorder=2)
            ax.vlines([vmin, vmax], yy - 0.050, yy + 0.050,
                      color=color, linewidth=0.9, alpha=0.58, zorder=2)
            ax.plot([q1, q3], [yy, yy], color=color, linewidth=2.35,
                    alpha=0.82, solid_capstyle="round", zorder=3)
            ax.vlines(med, yy - 0.105, yy + 0.105,
                      color=color, linewidth=1.55, alpha=0.95, zorder=4)
            jitter = rng.normal(0, 0.030, len(vals))
            ax.scatter(vals, np.full(len(vals), yy) + jitter, s=14,
                       marker=marker, color=color, alpha=0.36,
                       edgecolor="white", linewidth=0.25, zorder=5)
            ax.scatter([mean], [yy], s=56, marker=marker, color=color,
                       edgecolor="white", linewidth=0.85, zorder=6)
    ax.set_xlim(0.6, 18.4)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([1, 3, 5, 7, 10, 14, 18])
    ax.set_xlabel("Rank (lower = higher priority)")
    ax.set_yticks([y_positions[m] for m in methods])
    ax.set_yticklabels(methods)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.82)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=panel_colors["NAco"],
               markeredgecolor="white", markersize=6.2, label="NAco"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=panel_colors["CLA"],
               markeredgecolor="white", markersize=5.8, label="CLA"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.025), ncol=2, columnspacing=0.9,
              handletextpad=0.35)
    style_axes(ax)


def draw_panel_c_upset(ax: plt.Axes, df: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=spec, height_ratios=[2.2, 1.0], hspace=0.04
    )
    ax_bar = fig.add_subplot(subgs[0, 0])
    ax_mat = fig.add_subplot(subgs[1, 0], sharex=ax_bar)
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
    region_palette = {"NAco": COLORS["blue"], "CLA": COLORS["naco"]}
    combos = pd.DataFrame(combo_rows).sort_values(
        ["count", "Region"], ascending=[False, True]
    ).reset_index(drop=True)
    x = np.arange(len(combos))
    bar_colors = [region_palette[r] for r in combos["Region"]]
    ax_bar.bar(x, combos["count"], color=bar_colors,
               width=0.58, alpha=0.88, edgecolor="white", linewidth=0.7)
    for xi, (_, row) in zip(x, combos.iterrows()):
        ax_bar.text(xi, row["count"] + 0.38, str(int(row["count"])),
                    ha="center", va="bottom", fontsize=7.0,
                    fontweight="bold", color=bar_colors[int(xi)])
    ymax = max(8, float(combos["count"].max()) + 2.4)
    ax_bar.set_ylim(0, ymax)
    ax_bar.set_ylabel("Splits")
    ax_bar.set_yticks([0, 5, 10, 15])
    ax_bar.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.78)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    style_axes(ax_bar)
    ax_bar.spines["bottom"].set_visible(False)

    ydot = [2, 1, 0]
    for xi, (_, row) in zip(x, combos.iterrows()):
        pattern = row["pattern"]
        color = region_palette[row["Region"]]
        active_y = []
        for yi, active in zip(ydot, pattern):
            ax_mat.scatter([xi], [yi], s=44 if active else 26,
                           facecolor=color if active else "white",
                           edgecolor=color if active else COLORS["light_gray"],
                           linewidth=0.85, zorder=3)
            if active:
                active_y.append(yi)
        if len(active_y) >= 2:
            ax_mat.plot([xi, xi], [min(active_y), max(active_y)], color=color,
                        linewidth=1.45, alpha=0.82, zorder=2)
        ax_mat.plot([xi - 0.24, xi + 0.24], [-0.62, -0.62], color=color,
                    linewidth=2.4, solid_capstyle="round", clip_on=False)
        ax_mat.text(xi, -0.88, row["Region"], ha="center", va="top",
                    fontsize=6.7, fontweight="bold", color=color, clip_on=False)
    ax_mat.set_yticks(ydot)
    ax_mat.set_yticklabels([])
    ax_mat.set_ylim(-0.98, 2.48)
    ax_mat.set_xlim(-0.65, len(combos) - 0.35)
    for yi, label in zip(ydot, ["Consensus Top-3", "Fused Top-3", "GNN-lite Top-10"]):
        ax_mat.text(1.018, yi, label, ha="left", va="center",
                    fontsize=6.35, fontweight="bold", color=COLORS["ink"],
                    transform=ax_mat.get_yaxis_transform(), clip_on=False)
    ax_mat.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_mat.tick_params(axis="y", length=0)
    ax_mat.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.62)
    style_axes(ax_mat)
    ax_mat.spines["bottom"].set_visible(False)


def draw_panel_d_frequency_dotplot(ax: plt.Axes, df: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.18)
    methods = ["Consensus", "Fused", "GNN-lite"]
    rank_cols = [RANK_COL[m] for m in methods]
    categories = ["Top-3", "Top-10 only", "Not Top-10"]
    cat_colors = {
        "Top-3": COLORS["blue"],
        "Top-10 only": COLORS["naco"],
        "Not Top-10": COLORS["light_gray"],
    }

    def category(rank: float) -> str:
        if rank <= 3:
            return "Top-3"
        if rank <= 10:
            return "Top-10 only"
        return "Not Top-10"

    def draw_curve(axis: plt.Axes, x0: float, y0: float, x1: float, y1: float,
                   color: str, lw: float = 1.45, alpha: float = 0.20) -> None:
        dx = x1 - x0
        verts = [
            (x0, y0),
            (x0 + 0.48 * dx, y0),
            (x1 - 0.48 * dx, y1),
            (x1, y1),
        ]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                               MplPath.CURVE4, MplPath.CURVE4])
        axis.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                                 linewidth=lw, alpha=alpha, zorder=1,
                                 capstyle="round"))

    for idx, region in enumerate(SECOND_TIER):
        subax = fig.add_subplot(subgs[0, idx])
        sub = df[df["BrainRegion"].eq(region)].sort_values("fold").copy()
        y_offset = 0.35
        cat_by_method = {
            method: sub[RANK_COL[method]].map(category).tolist()
            for method in methods
        }
        y_positions: dict[str, dict[int, float]] = {}
        count_by_method: dict[str, dict[str, int]] = {}
        for method in methods:
            counts = {cat: cat_by_method[method].count(cat) for cat in categories}
            count_by_method[method] = counts
            bases = {}
            cursor = 0.0
            for cat in reversed(categories):
                bases[cat] = cursor
                cursor += counts[cat]
            pos = {}
            used = {cat: 0 for cat in categories}
            for fold, cat in zip(sub["fold"].tolist(), cat_by_method[method]):
                pos[int(fold)] = y_offset + bases[cat] + used[cat] + 0.5
                used[cat] += 1
            y_positions[method] = pos
        xloc = np.arange(len(methods), dtype=float)
        for _, row in sub.iterrows():
            fold = int(row["fold"])
            cats_for_fold = [category(float(row[col])) for col in rank_cols]
            for j in range(len(methods) - 1):
                draw_curve(
                    subax,
                    xloc[j] + 0.08,
                    y_positions[methods[j]][fold],
                    xloc[j + 1] - 0.08,
                    y_positions[methods[j + 1]][fold],
                    cat_colors[cats_for_fold[j]],
                )
        bar_w = 0.14
        for j, method in enumerate(methods):
            base = y_offset
            for cat in reversed(categories):
                count = count_by_method[method][cat]
                if count:
                    subax.add_patch(Rectangle(
                        (xloc[j] - bar_w / 2, base), bar_w, count,
                        facecolor=cat_colors[cat], edgecolor="white",
                        linewidth=0.65, alpha=0.95, zorder=3,
                    ))
                    if count >= 5:
                        color = "white" if cat != "Not Top-10" else COLORS["ink"]
                        subax.text(xloc[j], base + count / 2, str(count),
                                   ha="center", va="center", fontsize=6.3,
                                   fontweight="bold", color=color, zorder=4)
                base += count
        subax.text(0.02, 1.015, region, transform=subax.transAxes,
                   ha="left", va="bottom", fontsize=8.5, fontweight="bold",
                   color=REGION_COLORS.get(region, COLORS["ink"]),
                   clip_on=False)
        subax.set_xlim(-0.32, len(methods) - 0.68)
        subax.set_ylim(-0.35, 25.65)
        subax.set_xticks(xloc)
        subax.set_xticklabels(methods, rotation=18, ha="right")
        subax.set_yticks([0, 5, 10, 15, 20, 25])
        if idx == 0:
            subax.set_ylabel("Cross-validation splits")
        else:
            subax.set_yticklabels([])
        subax.grid(axis="y", color=COLORS["grid"], linewidth=0.50, alpha=0.65)
        style_axes(subax)
        if idx == 1:
            handles = [
                Line2D([0], [0], color=cat_colors["Top-3"], linewidth=3.2, label="Top-3"),
                Line2D([0], [0], color=cat_colors["Top-10 only"], linewidth=3.2, label="Top-10 only"),
                Line2D([0], [0], color=cat_colors["Not Top-10"], linewidth=3.2, label="Not Top-10"),
            ]
            subax.legend(handles=handles, frameon=False, loc="lower center",
                         bbox_to_anchor=(-0.10, 1.03), ncol=3,
                         columnspacing=0.70, handlelength=1.05,
                         handletextpad=0.35, fontsize=6.4)


def draw_panel_e_rank_shift(ax: plt.Axes, df: pd.DataFrame) -> None:
    methods = [
        ("Consensus", "consensus_rank", METHOD_COLORS["Consensus"]),
        ("Fused", "fused_rank", METHOD_COLORS["Fused"]),
        ("GNN-lite", "gnn_rank", METHOD_COLORS["GNN-lite"]),
    ]
    cutoffs = np.arange(1, 19)
    styles = {
        "NAco": {"linestyle": "-", "marker": "o", "alpha": 0.98},
        "CLA": {"linestyle": (0, (4, 2)), "marker": "D", "alpha": 0.90},
    }
    ax.axvspan(1, 3, color=COLORS["soft_green"], alpha=0.50, zorder=0)
    ax.axvspan(3, 10, color=COLORS["soft_blue"], alpha=0.36, zorder=0)
    for xline, label, color in [(3, "Top-3", COLORS["naco"]), (10, "Top-10", COLORS["dark_gray"])]:
        ax.axvline(xline, color=color, linewidth=0.85, linestyle="--",
                   alpha=0.55, zorder=1)
        ax.text(xline, 1.012, label, ha="center", va="bottom",
                fontsize=6.8, fontweight="bold", color=color,
                transform=ax.get_xaxis_transform(), clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": 0.12, "alpha": 0.82})
    for method, col, color in methods:
        for region in SECOND_TIER:
            vals = df[df["BrainRegion"].eq(region)][col].to_numpy(dtype=float)
            retained = np.array([(vals <= k).mean() for k in cutoffs])
            st = styles[region]
            ax.step(cutoffs, retained, where="post", color=color,
                    linewidth=2.05 if region == "NAco" else 1.70,
                    linestyle=st["linestyle"], alpha=st["alpha"], zorder=3)
            key_x = np.array([3, 10])
            key_y = np.array([(vals <= k).mean() for k in key_x])
            ax.scatter(key_x, key_y, s=32 if region == "NAco" else 28,
                       marker=st["marker"], color=color, edgecolor="white",
                       linewidth=0.75, zorder=4)
    ax.set_xlim(1, 19.2)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([1, 3, 5, 10, 14, 18])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("Cumulative retention frequency")
    ax.set_xlabel("Rank cutoff")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.82)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS["Consensus"], linewidth=2.0, label="Consensus"),
        Line2D([0], [0], color=METHOD_COLORS["Fused"], linewidth=2.0, label="Fused"),
        Line2D([0], [0], color=METHOD_COLORS["GNN-lite"], linewidth=2.0, label="GNN-lite"),
        Line2D([0], [0], color=COLORS["axis"], linestyle="-", marker="o",
               markeredgecolor="white", linewidth=1.3, markersize=4.8,
               label="NAco"),
        Line2D([0], [0], color=COLORS["axis"], linestyle=(0, (4, 2)), marker="D",
               markeredgecolor="white", linewidth=1.3, markersize=4.5,
               label="CLA"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.105), ncol=5,
              columnspacing=0.65, handlelength=1.1, handletextpad=0.30,
              fontsize=6.4)
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
    xvals = gaps["Tier 2 - Tier 1"].to_numpy(dtype=float)
    yvals = gaps["Supplementary - Tier 2"].to_numpy(dtype=float)
    xx = np.linspace(-0.8, 5.4, 180)
    yy = np.linspace(-1.0, 5.4, 180)
    gx, gy = np.meshgrid(xx, yy)
    sx = max(0.35, np.std(xvals, ddof=1) * 0.45)
    sy = max(0.35, np.std(yvals, ddof=1) * 0.45)
    density = np.zeros_like(gx)
    for xv, yv in zip(xvals, yvals):
        density += np.exp(-0.5 * (((gx - xv) / sx) ** 2 + ((gy - yv) / sy) ** 2))
    levels = np.linspace(density.max() * 0.15, density.max() * 0.96, 5)
    ax.contourf(gx, gy, density, levels=levels, colors=[COLORS["soft_green"]],
                alpha=0.58, zorder=0)
    ax.contour(gx, gy, density, levels=levels[1:], colors=[COLORS["naco"]],
               linewidths=0.78, alpha=0.58, zorder=1)
    ax.axvline(0, color=COLORS["axis"], linewidth=0.85, linestyle="--",
               alpha=0.55, zorder=2)
    ax.axhline(0, color=COLORS["axis"], linewidth=0.85, linestyle="--",
               alpha=0.55, zorder=2)
    ax.scatter(xvals, yvals, s=32, color=COLORS["naco"], alpha=0.68,
               edgecolor="white", linewidth=0.55, zorder=3)
    mx, my = float(np.mean(xvals)), float(np.mean(yvals))
    cx, cy = ci95(xvals)[1], ci95(yvals)[1]
    ax.errorbar(mx, my, xerr=cx, yerr=cy, fmt="D", markersize=7.0,
                color=COLORS["supp"], ecolor=COLORS["supp"], capsize=4.2,
                markeredgecolor="white", markeredgewidth=0.9,
                elinewidth=1.35, zorder=4)
    ax.text(mx + 0.22, my + 0.18, f"{mx:.2f}, {my:.2f}",
            ha="left", va="bottom", fontsize=7.5, fontweight="bold",
            color=COLORS["supp"],
            bbox={"facecolor": "white", "edgecolor": "none",
                  "pad": 0.18, "alpha": 0.88}, zorder=5)
    ax.set_xlabel("Tier 2 - Tier 1 fused-rank gap")
    ax.set_ylabel("Supplementary - Tier 2 gap")
    ax.set_xlim(-0.8, 5.45)
    ax.set_ylim(-1.0, 5.45)
    ax.set_xticks([-0.5, 0, 1, 2, 3, 4, 5])
    ax.set_yticks([-1, 0, 1, 2, 3, 4, 5])
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.80)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=COLORS["naco"], markeredgecolor="white",
               markersize=5.8, label="Split"),
        Line2D([0], [0], marker="D", color=COLORS["supp"],
               markerfacecolor=COLORS["supp"], markeredgecolor="white",
               markersize=6.0, label="Mean with 95% CI"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.52, 1.02), ncol=2, columnspacing=0.9,
              handletextpad=0.35, fontsize=6.8)
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
        ("panel_D_topk_frequency_mean_rank_dotplot", draw_panel_d_frequency_dotplot, (6.6, 3.25), (df,)),
        ("panel_E_consensus_fused_rank_shift", draw_panel_e_rank_shift, (4.8, 3.4), (df,)),
        ("panel_F_tier_gap_forest", draw_panel_g_tier_gap_forest, (4.6, 3.35), (df,)),
    ]
    for stem, drawer, size, args in panels:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax, *args)
        save_all(fig, stem)


def make_combined(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14.8, 9.2))
    gs = GridSpec(
        3, 24, figure=fig,
        height_ratios=[1.05, 1.05, 0.92],
        width_ratios=[1] * 24,
        wspace=1.18,
        hspace=0.55,
    )
    axes = [
        fig.add_subplot(gs[0, 0:14]),
        fig.add_subplot(gs[0, 14:24]),
        fig.add_subplot(gs[1, 0:8]),
        fig.add_subplot(gs[1, 8:24]),
        fig.add_subplot(gs[2, 0:12]),
        fig.add_subplot(gs[2, 12:24]),
    ]
    draw_panel_a_oncoprint(axes[0], df, summary)
    draw_panel_b_raincloud(axes[1], df)
    draw_panel_c_upset(axes[2], df)
    draw_panel_d_frequency_dotplot(axes[3], df)
    draw_panel_e_rank_shift(axes[4], df)
    draw_panel_g_tier_gap_forest(axes[5], df)
    fig.savefig(OUT / "figure_3_3_second_tier_bio_6panel_combined.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure_3_3_second_tier_bio_6panel_combined.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure_3_3_second_tier_bio_6panel_combined.svg",
                bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT / "figure_3_3_second_tier_bio_6panel_combined_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def zip_outputs() -> Path:
    zip_path = ROOT / "figure_3_3_second_tier_bio_6panel_300dpi_tiff.zip"
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
