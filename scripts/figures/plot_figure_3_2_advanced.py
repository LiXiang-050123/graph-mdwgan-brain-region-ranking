"""
Advanced Figure 3.2 panels for candidate brain-region prioritization.

Outputs:
- 8 individual panels as PNG/PDF/SVG
- A combined 8-panel figure as PNG/PDF/SVG
- A compact ranking summary CSV used by the panels
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch
from matplotlib.patches import PathPatch

from analysis_paths import (
    CODE_ROOT,
    CORRECTED_SECONDARY_RESULTS,
    PRIMARY_CV_RESULTS,
    SEED_RESULTS,
)


ROOT = CODE_ROOT
CV = PRIMARY_CV_RESULTS
OUT = ROOT / "figure_3_2_advanced"
OUT.mkdir(exist_ok=True)

REGIONS = [
    "ACC", "AM", "BLA", "CeA", "CLA", "CM", "DMS", "GP", "IL",
    "LH", "MS", "NAco", "NAsh", "OFC", "PL", "RSC", "VP", "VTA",
]
CANDIDATES = ["ACC", "VTA", "NAco", "CLA", "BLA", "CeA", "DMS"]
TIERS = {
    "ACC": "Tier 1", "VTA": "Tier 1",
    "NAco": "Tier 2", "CLA": "Tier 2",
    "BLA": "Supplementary", "CeA": "Supplementary", "DMS": "Supplementary",
}

COLORS = {
    "ink": "#0B2A4A",
    "blue": "#0B559F",
    "lake": "#3DA5D9",
    "emerald": "#1FA463",
    "green": "#56B586",
    "ochre": "#B8752A",
    "rose": "#C75B59",
    "gray": "#8A8F98",
    "light_gray": "#DDE3EA",
    "pale": "#F3F6F8",
    "axis": "#202020",
}

TIER_COLORS = {
    "Tier 1": COLORS["ink"],
    "Tier 2": COLORS["emerald"],
    "Supplementary": COLORS["ochre"],
    "Other": COLORS["gray"],
}

RANK_CMAP = LinearSegmentedColormap.from_list(
    "rank_map",
    ["#08304A", "#146C7C", "#65A9B8", "#D2E4EA", "#F4F6F7"],
)
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_map",
    ["#EEF3F7", "#BBD5E0", "#6DA8B6", "#19767E", "#083D4A"],
)


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.4,
        "ytick.major.size": 3.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    })


def style_axes(ax: plt.Axes, top_right_off: bool = True) -> None:
    if top_right_off:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="both", colors="black", labelcolor="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Arial")


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


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
        fused = pd.read_csv(fusion_path)
        fused = fused.rename(columns={
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


def ranking_summary(df: pd.DataFrame) -> pd.DataFrame:
    def top_count(col: str, k: int):
        return lambda s: int((s <= k).sum())

    summary = df.groupby("BrainRegion").agg(
        mean_consensus_rank=("consensus_rank", "mean"),
        sd_consensus_rank=("consensus_rank", "std"),
        mean_fused_rank=("fused_rank", "mean"),
        sd_fused_rank=("fused_rank", "std"),
        mean_gnn_rank=("gnn_rank", "mean"),
        sd_gnn_rank=("gnn_rank", "std"),
        top1_fused=("fused_rank", top_count("fused_rank", 1)),
        top3_fused=("fused_rank", top_count("fused_rank", 3)),
        top5_fused=("fused_rank", top_count("fused_rank", 5)),
        top10_fused=("fused_rank", top_count("fused_rank", 10)),
        top1_consensus=("consensus_rank", top_count("consensus_rank", 1)),
        top3_consensus=("consensus_rank", top_count("consensus_rank", 3)),
        top5_consensus=("consensus_rank", top_count("consensus_rank", 5)),
        top10_consensus=("consensus_rank", top_count("consensus_rank", 10)),
        top1_gnn=("gnn_rank", top_count("gnn_rank", 1)),
        top3_gnn=("gnn_rank", top_count("gnn_rank", 3)),
        top5_gnn=("gnn_rank", top_count("gnn_rank", 5)),
        top10_gnn=("gnn_rank", top_count("gnn_rank", 10)),
    ).reset_index()
    summary["tier"] = summary["BrainRegion"].map(TIERS).fillna("Other")
    summary = summary.sort_values("mean_fused_rank")
    summary.to_csv(OUT / "figure_3_2_ranking_summary.csv", index=False)
    return summary


def panel_rank_heatmap(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()

    order = summary["BrainRegion"].tolist()
    specs = [("Consensus rank", "consensus_rank"), ("Fused rank", "fused_rank")]
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[1, 1, 0.045], wspace=0.12)
    images = []
    for idx, (label, col) in enumerate(specs):
        subax = fig.add_subplot(subgs[0, idx])
        pivot = df.pivot(index="BrainRegion", columns="fold", values=col).loc[order]
        im = subax.imshow(pivot.to_numpy(), cmap=RANK_CMAP, vmin=1, vmax=18, aspect="auto")
        images.append(im)
        subax.text(0.02, 1.035, label, transform=subax.transAxes, ha="left", va="bottom",
                   fontsize=8.2, fontweight="bold", color=COLORS["ink"])
        subax.set_yticks(np.arange(len(order)))
        subax.set_yticklabels(order if idx == 0 else [])
        subax.set_xticks([0, 4, 9, 14, 19, 24])
        subax.set_xticklabels(["1", "5", "10", "15", "20", "25"])
        subax.set_xlabel("Cross-validation split")
        if idx == 0:
            subax.set_ylabel("Brain region")
            for i, region in enumerate(order):
                tier = TIERS.get(region, "Other")
                subax.add_patch(plt.Rectangle(
                    (-1.10, i - 0.30), 0.22, 0.60,
                    transform=subax.transData,
                    facecolor=TIER_COLORS[tier],
                    edgecolor="none",
                    clip_on=False,
                    alpha=0.95,
                ))
        subax.set_xticks(np.arange(-0.5, pivot.shape[1], 1), minor=True)
        subax.set_yticks(np.arange(-0.5, pivot.shape[0], 1), minor=True)
        subax.grid(which="minor", color="white", linewidth=0.32, alpha=0.72)
        subax.tick_params(which="minor", bottom=False, left=False)
        subax.tick_params(axis="both", labelsize=6.5, pad=1.5)
        for spine in subax.spines.values():
            spine.set_visible(False)
        style_axes(subax, top_right_off=False)
    cax = fig.add_subplot(subgs[0, 2])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_ticks([1, 5, 10, 18])
    cbar.ax.invert_yaxis()
    cbar.set_label("Rank\n1 = best", fontsize=6.7, fontweight="bold", labelpad=3)
    cbar.ax.tick_params(labelsize=6.5, width=0.7, length=2.4, pad=1.2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Arial")
    cbar.outline.set_linewidth(0.6)


def panel_method_matrix(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    columns = [
        "score_ANOVA", "score_LASSO", "score_RF", "score_RFE", "score_MI",
        "ConsensusScore", "GNNlite_coef_norm", "ConsensusPlusGNN",
    ]
    labels = ["ANOVA", "LASSO", "RF", "RFE-SVM", "MI", "Consensus", "GNN-lite", "Fused"]
    mat = df[df["BrainRegion"].isin(CANDIDATES)].groupby("BrainRegion")[columns].mean().loc[CANDIDATES]
    im = ax.imshow(mat.to_numpy(), cmap=SCORE_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(np.arange(len(CANDIDATES)))
    ax.set_yticklabels(CANDIDATES)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for yy in range(mat.shape[0]):
        for xx in range(mat.shape[1]):
            val = mat.iat[yy, xx]
            color = "white" if val >= 0.70 else COLORS["ink"]
            ax.text(xx, yy, f"{val:.2f}", ha="center", va="center", fontsize=6.7,
                    fontweight="bold", color=color)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
    cbar.set_label("Direction-aligned\npriority score", fontsize=6.7,
                   fontweight="bold", labelpad=3)
    cbar.set_ticks([0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.ax.tick_params(labelsize=6.5, width=0.7, length=2.4, pad=1.2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Arial")
    cbar.outline.set_linewidth(0.6)
    style_axes(ax, top_right_off=False)


def panel_topk_bubble(ax: plt.Axes, summary: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()

    rows = CANDIDATES
    groups = [
        ("Consensus", ["top1_consensus", "top3_consensus", "top5_consensus", "top10_consensus"], COLORS["blue"]),
        ("Fused", ["top1_fused", "top3_fused", "top5_fused", "top10_fused"], COLORS["emerald"]),
        ("GNN-lite", ["top1_gnn", "top3_gnn", "top5_gnn", "top10_gnn"], COLORS["ochre"]),
    ]
    k_labels = ["Top-1", "Top-3", "Top-5", "Top-10"]
    sub = summary.set_index("BrainRegion").loc[rows]
    subgs = GridSpecFromSubplotSpec(
        1, 4, subplot_spec=spec, width_ratios=[1, 1, 1, 0.055], wspace=0.08
    )
    for gi, (group_label, columns, color) in enumerate(groups):
        subax = fig.add_subplot(subgs[0, gi])
        mat = sub[columns].to_numpy(dtype=float)
        freq = mat / 25.0
        image = SCORE_CMAP(freq)[..., :3]
        subax.imshow(image, aspect="auto", vmin=0, vmax=1)
        for yy, region in enumerate(rows):
            for xx, col in enumerate(columns):
                val = int(sub.loc[region, col])
                freq_label = int(round(val / 25 * 100))
                text_color = "white" if freq_label >= 80 else COLORS["ink"]
                subax.text(xx, yy, f"{freq_label}%", ha="center", va="center",
                           fontsize=6.55, fontweight="bold", color=text_color)
        subax.text(0.5, 1.045, group_label, transform=subax.transAxes,
                   ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                   color=COLORS["ink"])
        subax.set_xticks(np.arange(len(k_labels)))
        subax.set_xticklabels(k_labels, rotation=35, ha="right")
        subax.set_yticks(np.arange(len(rows)))
        subax.set_yticklabels(rows if gi == 0 else [])
        subax.set_xticks(np.arange(-0.5, len(k_labels), 1), minor=True)
        subax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        subax.grid(which="minor", color="white", linewidth=1.05)
        subax.tick_params(which="minor", bottom=False, left=False)
        subax.tick_params(axis="both", labelsize=7.0, pad=1.6)
        for spine in subax.spines.values():
            spine.set_visible(False)
        style_axes(subax, top_right_off=False)
    cax = fig.add_subplot(subgs[0, 3])
    sm = plt.cm.ScalarMappable(cmap=SCORE_CMAP, norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Selection\nfrequency (%)", fontsize=6.7,
                   fontweight="bold", labelpad=3)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=6.5, width=0.7, length=2.4, pad=1.2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Arial")
    cbar.outline.set_linewidth(0.6)


def panel_rank_shift(ax: plt.Axes, summary: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[3.15, 1.0], wspace=0.08)
    ax = fig.add_subplot(subgs[0, 0])
    delta_ax = fig.add_subplot(subgs[0, 1], sharey=ax)

    sub = summary.sort_values("mean_fused_rank", ascending=True).reset_index(drop=True)
    y = np.arange(len(sub))
    candidate_gray = "#666D76"
    other_gray = "#A5ABB3"
    link_gray = "#C8CED5"
    ax.axvspan(0.5, 3.5, color="#DDE3EA", alpha=0.28, linewidth=0, zorder=0)
    ax.axvspan(3.5, 10.5, color="#F3F6F8", alpha=0.78, linewidth=0, zorder=0)
    for yi, row in sub.iterrows():
        region = row["BrainRegion"]
        is_candidate = region in CANDIDATES
        color = candidate_gray if is_candidate else other_gray
        consensus = float(row["mean_consensus_rank"])
        fused = float(row["mean_fused_rank"])
        ax.plot([consensus, fused], [yi, yi], color=link_gray,
                linewidth=1.0 if is_candidate else 0.75,
                alpha=0.86 if is_candidate else 0.55, solid_capstyle="round", zorder=1)
        ax.scatter(consensus, yi, s=30 if is_candidate else 18,
                   marker="o", facecolor="white", edgecolor=color,
                   linewidth=1.0 if is_candidate else 0.7, zorder=3)
        ax.scatter(fused, yi, s=36 if is_candidate else 20,
                   marker="D", facecolor=color, edgecolor="white",
                   linewidth=0.65, alpha=0.95, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["BrainRegion"].tolist())
    for label, region in zip(ax.get_yticklabels(), sub["BrainRegion"].tolist()):
        label.set_color(COLORS["axis"])
        label.set_fontweight("bold" if region in CANDIDATES else "normal")
    scheme_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=4.6,
               markerfacecolor="white", markeredgecolor=COLORS["axis"],
               label="Consensus"),
        Line2D([0], [0], marker="D", linestyle="", markersize=4.6,
               markerfacecolor=COLORS["axis"], markeredgecolor="white",
               label="Fused"),
    ]
    ax.legend(handles=scheme_handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.56, 1.075), ncol=2, handletextpad=0.42,
              columnspacing=1.05, borderaxespad=0.0, fontsize=8.0)
    ax.set_xlabel("Mean rank (lower = higher priority)")
    ax.set_ylabel("")
    ax.set_xlim(0.5, 18.5)
    ax.set_xticks([1, 5, 10, 15, 18])
    ax.set_ylim(-1.15, len(sub) - 0.25)
    ax.invert_yaxis()
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.6, alpha=0.75)
    ax.grid(axis="y", color=COLORS["pale"], linewidth=0.4, alpha=0.70)
    ax.text(2.0, -0.78, "Top-3", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#7F8791")
    ax.text(7.0, -0.78, "Top-10", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#8A8F98")
    style_axes(ax)

    delta = sub["mean_fused_rank"].to_numpy(dtype=float) - sub["mean_consensus_rank"].to_numpy(dtype=float)
    for yi, row in sub.iterrows():
        region = row["BrainRegion"]
        is_candidate = region in CANDIDATES
        color = candidate_gray if is_candidate else other_gray
        value = float(row["mean_fused_rank"] - row["mean_consensus_rank"])
        delta_ax.plot([0, value], [yi, yi], color=color,
                      linewidth=1.15 if is_candidate else 0.75,
                      alpha=0.82 if is_candidate else 0.52, solid_capstyle="round", zorder=2)
        delta_ax.scatter(value, yi, s=22 if is_candidate else 12,
                         marker="s", color=color, edgecolor="white",
                         linewidth=0.45, zorder=3)
    delta_ax.axvline(0, color=COLORS["axis"], linewidth=0.65, alpha=0.72)
    delta_ax.set_xlim(-3.4, 5.4)
    delta_ax.set_xticks([-3, 0, 3])
    delta_ax.set_xlabel("Delta rank\nFused - consensus", fontsize=7.1, labelpad=2)
    delta_ax.tick_params(axis="y", left=False, labelleft=False)
    delta_ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.55, alpha=0.72)
    delta_ax.grid(axis="y", color=COLORS["pale"], linewidth=0.4, alpha=0.70)
    style_axes(delta_ax)


def panel_rank_lollipop(ax: plt.Axes, summary: pd.DataFrame) -> None:
    order = summary["BrainRegion"].tolist()[::-1]
    sub = summary.set_index("BrainRegion").loc[order]
    y = np.arange(len(order))
    for yi, region in enumerate(order):
        tier = sub.loc[region, "tier"]
        color = TIER_COLORS[tier]
        mean = sub.loc[region, "mean_fused_rank"]
        sd = sub.loc[region, "sd_fused_rank"]
        rate = sub.loc[region, "top10_fused"] / 25
        ax.hlines(yi, 18, mean, color=COLORS["light_gray"], linewidth=1.0, zorder=1)
        ax.errorbar(mean, yi, xerr=sd, fmt="none", color=COLORS["axis"], capsize=2, elinewidth=0.8, zorder=2)
        ax.scatter(mean, yi, s=35 + 150 * rate, color=color, edgecolor="white", linewidth=0.65, zorder=3)
        if region in CANDIDATES:
            ax.text(mean - 0.25, yi, f"{mean:.2f}", ha="right", va="center",
                    fontsize=7.4, fontweight="bold", color=COLORS["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Mean fused rank")
    ax.set_xlim(18.5, 0.5)
    ax.grid(axis="x", color=COLORS["light_gray"], linewidth=0.55, alpha=0.70)
    handles = [Patch(facecolor=TIER_COLORS[k], label=k) for k in ["Tier 1", "Tier 2", "Supplementary", "Other"]]
    ax.legend(handles=handles, frameon=False, loc="lower right", ncol=2, handlelength=1.0, columnspacing=0.8)
    style_axes(ax)


def load_real_matrix() -> pd.DataFrame:
    seed_dir = SEED_RESULTS / "seed_20250829" / "ours"
    train = pd.read_csv(seed_dir / "train_augmented_x8.csv")
    test = pd.read_csv(seed_dir / "test_original.csv")
    real = pd.concat([train[train["Origin"].eq("RealTrain")], test], ignore_index=True)
    return real[REGIONS].astype(float)


def panel_candidate_network(ax: plt.Axes, summary: pd.DataFrame) -> None:
    real = load_real_matrix()
    corr = np.log1p(real[CANDIDATES]).corr(method="spearman")
    pos = {
        "ACC": np.array([-0.95, 0.38]),
        "CLA": np.array([-0.18, 0.82]),
        "VTA": np.array([-0.86, -0.46]),
        "NAco": np.array([-0.02, -0.12]),
        "BLA": np.array([0.72, 0.36]),
        "CeA": np.array([0.92, -0.30]),
        "DMS": np.array([0.24, -0.78]),
    }
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) < 0.45:
                continue
            p1, p2 = pos[a], pos[b]
            color = COLORS["emerald"] if r >= 0 else COLORS["rose"]
            style = "-" if r >= 0 else "--"
            mid = (p1 + p2) / 2
            normal = np.array([-(p2 - p1)[1], (p2 - p1)[0]])
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            curve = mid + normal * 0.12 * np.sign(i - 2.4)
            verts = [tuple(p1), tuple(curve), tuple(p2)]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=color,
                                   linestyle=style, linewidth=0.6 + 2.5 * abs(r),
                                   alpha=0.18 + 0.50 * abs(r), capstyle="round", zorder=1))
    sub = summary.set_index("BrainRegion")
    for region in CANDIDATES:
        tier = TIERS[region]
        p = pos[region]
        size = 300 + 22 * sub.loc[region, "top3_fused"]
        ax.scatter(p[0], p[1], s=size * 1.24, color=TIER_COLORS[tier], alpha=0.16,
                   edgecolor="none", zorder=2)
        ax.scatter(p[0], p[1], s=size, color=TIER_COLORS[tier], edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter(p[0], p[1], s=size * 0.78, color="none", edgecolor="white", linewidth=0.45, alpha=0.75, zorder=4)
        ax.text(p[0], p[1], region, ha="center", va="center", color="white", fontsize=8.4, fontweight="bold", zorder=4)
    handles = [
        Line2D([0], [0], color=COLORS["emerald"], linewidth=2.0, label="Positive rho"),
        Line2D([0], [0], color=COLORS["rose"], linewidth=2.0, linestyle="--", label="Negative rho"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.50, -0.08),
              ncol=2, handlelength=1.7, columnspacing=1.0)
    ax.set_xlim(-1.20, 1.18)
    ax.set_ylim(-1.00, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_performance_topk(ax: plt.Axes) -> None:
    perf = pd.read_csv(CV / "auc_k_mean_sd.csv")
    sub = perf[(perf["method"].eq("Ours")) & (perf["scheme"].isin(["consensus", "fusion"]))]
    xmap = {1: 0, 3: 1, 5: 2, 10: 3}
    styles = {
        "consensus": (COLORS["blue"], "o", "Consensus"),
        "fusion": (COLORS["emerald"], "s", "Fused"),
    }
    for scheme, (color, marker, label) in styles.items():
        s = sub[sub["scheme"].eq(scheme)].sort_values("k")
        x = [xmap[int(k)] for k in s["k"]]
        ax.errorbar(x, s["auc_mean"], yerr=s["auc_sd"], color=color, marker=marker,
                    linewidth=1.8, markersize=5.0, capsize=3, label=f"{label} AUC")
        ax.plot(x, s["auprc_mean"], color=color, marker=marker, linestyle="--",
                linewidth=1.2, markersize=4.4, alpha=0.78, label=f"{label} AUPRC")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Top-1", "Top-3", "Top-5", "Top-10"])
    ax.set_ylabel("Validation metric")
    ax.set_ylim(0.70, 1.03)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.6, alpha=0.75)
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.7)
    style_axes(ax)


def panel_random_baseline(ax: plt.Axes) -> None:
    rand = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "random_roi_baseline.csv")
    rand = rand[rand["method"].eq("Ours")]
    perf = pd.read_csv(CV / "auc_k_mean_sd.csv")
    obs = perf[(perf["method"].eq("Ours")) & (perf["scheme"].eq("fusion"))].sort_values("k")
    k_values = [1, 3, 5, 10]
    positions = np.arange(len(k_values))
    parts = ax.violinplot([rand[rand["k"].eq(k)]["auc"].to_numpy() for k in k_values],
                          positions=positions, widths=0.58, showmeans=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(COLORS["light_gray"])
        body.set_edgecolor(COLORS["gray"])
        body.set_alpha(0.62)
        body.set_linewidth(0.8)
    rng = np.random.default_rng(2032)
    for i, k in enumerate(k_values):
        vals = rand[rand["k"].eq(k)]["auc"].sample(n=120, random_state=100 + k).to_numpy()
        ax.scatter(np.full_like(vals, i, dtype=float) + rng.normal(0, 0.035, len(vals)), vals,
                   s=4, color=COLORS["gray"], alpha=0.26, linewidths=0)
    ax.errorbar(positions, obs["auc_mean"], yerr=obs["auc_sd"], color=COLORS["emerald"],
                marker="D", linewidth=1.7, markersize=5.2, capsize=3, label="Prioritized regions")
    for i, row in enumerate(obs.itertuples()):
        ax.text(i + 0.09, row.auc_mean + 0.025, f"{row.auc_mean:.3f}",
                fontsize=7.6, fontweight="bold", color=COLORS["ink"], ha="left", va="bottom")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Top-{k}" for k in k_values])
    ax.set_ylabel("Validation AUC")
    ax.set_ylim(0.18, 1.04)
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.55, alpha=0.70)
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)


def make_figures() -> None:
    df = load_fold_rankings()
    summary = ranking_summary(df)
    panels = [
        ("advanced_panel_A_fused_rank_heatmap", panel_rank_heatmap, (5.2, 4.0), (df, summary)),
        ("advanced_panel_B_method_score_matrix", panel_method_matrix, (5.2, 4.0), (df, summary)),
        ("advanced_panel_C_topk_frequency_bubbles", panel_topk_bubble, (5.5, 3.2), (summary,)),
        ("advanced_panel_D_rank_shift", panel_rank_shift, (4.8, 4.2), (summary,)),
        ("advanced_panel_E_final_hierarchy", panel_rank_lollipop, (4.8, 4.2), (summary,)),
        ("advanced_panel_F_candidate_network", panel_candidate_network, (4.2, 3.8), (summary,)),
        ("advanced_panel_G_topk_performance", panel_performance_topk, (4.8, 3.2), tuple()),
        ("advanced_panel_H_random_baseline", panel_random_baseline, (4.8, 3.2), tuple()),
    ]
    for stem, func, size, args in panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax, *args)
        save_all(fig, stem)

    fig = plt.figure(figsize=(14.2, 14.0))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.18, 1.00, 1.16, 1.02],
                  width_ratios=[1.03, 1.0], hspace=0.72, wspace=0.34)
    axes = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]
    funcs = [
        (panel_rank_heatmap, (df, summary)),
        (panel_method_matrix, (df, summary)),
        (panel_topk_bubble, (summary,)),
        (panel_rank_shift, (summary,)),
        (panel_rank_lollipop, (summary,)),
        (panel_candidate_network, (summary,)),
        (panel_performance_topk, tuple()),
        (panel_random_baseline, tuple()),
    ]
    for ax, (func, args) in zip(axes, funcs):
        func(ax, *args)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_3_2_advanced_combined.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    make_figures()
    print(f"Saved Figure 3.2 advanced panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
