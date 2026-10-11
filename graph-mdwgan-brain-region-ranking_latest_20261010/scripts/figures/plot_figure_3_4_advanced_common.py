from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle, Wedge

import plot_figure_3_3_second_tier_bio_6panel as base
from analysis_paths import CODE_ROOT


ROOT = CODE_ROOT
OUT = ROOT / "figure_3_4_advanced_common_300dpi"
OUT.mkdir(exist_ok=True)

REGIONS = ["BLA", "CeA", "DMS"]
METHODS = [
    ("Consensus", "consensus_rank", "#2E86C1"),
    ("Fused", "fused_rank", "#22A884"),
    ("GNN-lite", "gnn_rank", "#C9843B"),
]
COLORS = {
    "ink": "#102A43",
    "axis": "#202020",
    "grid": "#E3E8EF",
    "soft": "#F4F7F9",
    "soft_orange": "#FBF2EA",
    "top3": "#2E86C1",
    "top10": "#22A884",
    "miss": "#C9D2DC",
    "supp": "#C9843B",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    })


def load_data() -> pd.DataFrame:
    df = base.load_fold_rankings()
    return df[df["BrainRegion"].isin(REGIONS)].copy()


def style_axes(ax: plt.Axes, top_right_off: bool = True) -> None:
    if top_right_off:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="both", colors="black", labelcolor="black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Arial")
        tick.set_fontweight("bold")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.10, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=13.5, fontweight="bold", color="black", clip_on=False)


def smooth_hist(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bins = np.arange(0.5, 18.6, 1.0)
    hist, edges = np.histogram(vals, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    y = np.interp(grid, centers, hist, left=0, right=0)
    kernel_x = np.linspace(-2.4, 2.4, 19)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def ci95(vals: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return float(np.mean(vals)), 0.0
    return float(np.mean(vals)), float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)))


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


def panel_a_ridgeline(ax: plt.Axes, df: pd.DataFrame) -> None:
    grid = np.linspace(1, 18, 240)
    offsets = []
    labels = []
    colors = []
    idx = 0
    for region in REGIONS:
        for name, col, color in METHODS:
            offsets.append(idx)
            labels.append(f"{region}  {name}")
            colors.append(color)
            vals = df[df["BrainRegion"].eq(region)][col].to_numpy(dtype=float)
            dens = smooth_hist(vals, grid)
            dens = dens / max(dens.max(), 1e-6) * 0.72
            y0 = idx
            ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.24, linewidth=0)
            ax.plot(grid, y0 + dens, color=color, linewidth=1.35)
            med = np.median(vals)
            ax.vlines(med, y0, y0 + 0.70, color=color, linewidth=1.0, alpha=0.82)
            ax.text(
                med + 0.18, y0 + 0.88, f"{med:.0f}", ha="left", va="center",
                fontsize=6.7, fontweight="bold", color=color, clip_on=False,
            )
            idx += 1
        idx += 0.35
    ax.axvspan(0.5, 3.5, color="#EFF8F5", alpha=0.58, zorder=0)
    ax.axvspan(3.5, 10.5, color="#EEF5FA", alpha=0.42, zorder=0)
    ax.set_xlim(1, 18)
    ax.set_ylim(-0.15, offsets[-1] + 1.18)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rank across splits")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.80)
    style_axes(ax)


def panel_b_estimation(ax: plt.Axes, df: pd.DataFrame) -> None:
    flow = df[["consensus_rank", "fused_rank", "gnn_rank"]].copy()
    flow["Consensus"] = rank_bin(flow["consensus_rank"])
    flow["Fused"] = rank_bin(flow["fused_rank"])
    flow["GNN-lite"] = rank_bin(flow["gnn_rank"])
    cats = ["Top-3", "Top-10 only", ">Top-10"]
    cat_colors = {"Top-3": COLORS["top3"], "Top-10 only": COLORS["top10"], ">Top-10": COLORS["miss"]}
    stages = ["Consensus", "Fused", "GNN-lite"]
    x_stage = {"Consensus": 0.09, "Fused": 0.49, "GNN-lite": 0.89}
    bar_w = 0.075
    total = float(len(flow))
    gap = 0.033

    def intervals(totals: pd.Series) -> dict[str, tuple[float, float]]:
        pos = {}
        y = 1.0
        for cat in cats:
            h = (totals[cat] / total) * (1 - gap * (len(cats) - 1))
            pos[cat] = (y - h, y)
            y -= h + gap
        return pos

    totals_by_stage = {stage: flow[stage].value_counts().reindex(cats, fill_value=0) for stage in stages}
    intervals_by_stage = {stage: intervals(totals_by_stage[stage]) for stage in stages}
    for left_stage, right_stage in zip(stages[:-1], stages[1:]):
        pair_counts = flow.groupby([left_stage, right_stage], observed=False).size().reset_index(name="n")
        left_cursor = {k: v[0] for k, v in intervals_by_stage[left_stage].items()}
        right_cursor = {k: v[0] for k, v in intervals_by_stage[right_stage].items()}
        x0, x1 = x_stage[left_stage] + bar_w, x_stage[right_stage]
        for _, row in pair_counts.sort_values([left_stage, right_stage]).iterrows():
            src, dst, n = str(row[left_stage]), str(row[right_stage]), int(row["n"])
            if n == 0:
                continue
            h = (n / total) * (1 - gap * (len(cats) - 1))
            y0s, y1s = left_cursor[src], left_cursor[src] + h
            y0t, y1t = right_cursor[dst], right_cursor[dst] + h
            left_cursor[src] += h
            right_cursor[dst] += h
            xm0 = x0 + (x1 - x0) * 0.42
            xm1 = x0 + (x1 - x0) * 0.58
            verts = [
                (x0, y0s), (xm0, y0s), (xm1, y0t), (x1, y0t),
                (x1, y1t), (xm1, y1t), (xm0, y1s), (x0, y1s), (x0, y0s),
            ]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=cat_colors[dst],
                                   edgecolor="white", linewidth=0.28, alpha=0.42))
            if n >= 6:
                ax.text((x0 + x1) / 2, (y0s + y1s + y0t + y1t) / 4, f"{n}",
                        ha="center", va="center", fontsize=5.8, fontweight="bold",
                        color=COLORS["ink"], zorder=5)
    for stage in stages:
        xpos = x_stage[stage]
        ax.text(xpos + bar_w / 2, 1.055, stage, ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color=COLORS["ink"])
        for cat in cats:
            y0, y1 = intervals_by_stage[stage][cat]
            ax.add_patch(Rectangle((xpos, y0), bar_w, y1 - y0, facecolor=cat_colors[cat],
                                   edgecolor="white", linewidth=0.7, alpha=0.92))
            ax.text(xpos + bar_w / 2, (y0 + y1) / 2, f"{int(totals_by_stage[stage][cat])}",
                    ha="center", va="center", fontsize=6.4, fontweight="bold",
                    color=("white" if cat != ">Top-10" else COLORS["ink"]))
            if stage == "Consensus":
                ax.text(xpos - 0.018, (y0 + y1) / 2, cat, ha="right", va="center",
                        fontsize=6.3, fontweight="bold", color=COLORS["ink"])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.09)
    ax.set_axis_off()


def count_states(vals: np.ndarray) -> tuple[int, int, int]:
    top3 = int(np.sum(vals <= 3))
    top10 = int(np.sum((vals > 3) & (vals <= 10)))
    miss = int(np.sum(vals > 10))
    return top3, top10, miss


def rank_bin(vals: pd.Series | np.ndarray) -> pd.Series:
    vals = pd.Series(vals)
    return pd.cut(vals, bins=[0, 3, 10, 99],
                  labels=["Top-3", "Top-10 only", ">Top-10"],
                  include_lowest=True)


def rank_status_value(rank: float) -> int:
    if rank <= 3:
        return 2
    if rank <= 10:
        return 1
    return 0


def panel_c_stacked(ax: plt.Axes, df: pd.DataFrame) -> None:
    rows = [(region, method, col, color) for region in REGIONS for method, col, color in METHODS]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(0.5, 3.5, color="#EFF8F5", alpha=0.56, zorder=0)
    ax.axvspan(3.5, 10.5, color="#EEF5FA", alpha=0.40, zorder=0)
    for yi, (region, method, col, color) in zip(y, rows):
        vals = df[df["BrainRegion"].eq(region)][col].to_numpy(dtype=int)
        counts = pd.Series(vals).value_counts().sort_index()
        for rank, count in counts.items():
            offsets = np.linspace(-0.21, 0.21, count) if count > 1 else np.array([0.0])
            sizes = np.linspace(17, 25, count)
            ax.scatter(
                np.full(count, rank),
                yi + offsets,
                s=sizes,
                color=color,
                alpha=0.58,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        median = float(np.median(vals))
        mean = float(np.mean(vals))
        ax.plot([median, median], [yi - 0.34, yi + 0.34], color=COLORS["ink"], linewidth=1.25, zorder=4)
        ax.scatter([mean], [yi], marker="D", s=34, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            18.35, yi, f"{mean:.2f}", ha="right", va="center",
            fontsize=6.3, fontweight="bold", color=color
        )
    for sep in [5.5, 2.5]:
        ax.axhline(sep, color=COLORS["grid"], linewidth=0.75)
    ax.set_xlim(0.5, 18.5)
    ax.set_xticks([1, 3, 5, 10, 15, 18])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  {m}" for r, m, _, _ in rows])
    ax.set_xlabel("Rank frequency across 25 splits")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777",
               markeredgecolor="white", markersize=5.0, label="Split count"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["ink"],
               markeredgecolor="white", markersize=5.0, label="Mean rank"),
        Line2D([0], [0], color=COLORS["ink"], linewidth=1.2, label="Median rank"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, columnspacing=0.8, handletextpad=0.35)
    style_axes(ax)


def panel_d_quantile_interval(ax: plt.Axes, df: pd.DataFrame) -> None:
    rows = [(region, method, col, color) for region in REGIONS for method, col, color in METHODS]
    y = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(17)
    for yi, (region, method, col, color) in zip(y, rows):
        vals = df[df["BrainRegion"].eq(region)][col].to_numpy(float)
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        p15, p85 = np.percentile(vals, [15, 85])
        ax.hlines(yi, p5, p95, color=color, linewidth=0.9, alpha=0.70, zorder=1)
        ax.hlines(yi, p15, p85, color=color, linewidth=8.5, alpha=0.13, zorder=1,
                  capstyle="round")
        ax.hlines(yi, p25, p75, color=color, linewidth=10.5, alpha=0.50, zorder=2,
                  capstyle="round")
        ax.vlines(p50, yi - 0.35, yi + 0.35, color=COLORS["ink"], linewidth=1.0, zorder=4)
        jitter = rng.normal(0, 0.045, len(vals))
        ax.scatter(vals, yi + jitter, s=11, color=color, alpha=0.40,
                   edgecolor="white", linewidth=0.25, zorder=3)
        ax.text(p50 + 0.34, yi + 0.10, f"{p50:.0f}", ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=color, zorder=6,
                path_effects=[pe.withStroke(linewidth=1.35, foreground="white")])
    for sep in [5.5, 2.5]:
        ax.axhline(sep, color=COLORS["grid"], linewidth=0.75)
    ax.axvspan(0.5, 3.5, color="#EFF8F5", alpha=0.42, zorder=0)
    ax.axvspan(3.5, 10.5, color="#EEF5FA", alpha=0.32, zorder=0)
    ax.set_xlim(0.5, 18.5)
    ax.set_xticks([1, 3, 5, 10, 15, 18])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  {m}" for r, m, _, _ in rows])
    ax.set_xlabel("Rank distribution")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    style_axes(ax)


def panel_e_co_retention(ax: plt.Axes, df: pd.DataFrame) -> None:
    rows = []
    matrix = []
    method_colors = {name: color for name, _, color in METHODS}
    region_colors = {"BLA": COLORS["supp"], "CeA": COLORS["top3"], "DMS": COLORS["top10"]}
    method_labels = {"Consensus": "Cons.", "Fused": "Fused", "GNN-lite": "GNN-lite"}
    for region in REGIONS:
        for method, col, color in METHODS:
            rows.append(f"{region}  {method_labels[method]}")
            vals = df[df["BrainRegion"].eq(region)].sort_values("fold")[col].to_numpy(float)
            matrix.append(vals)
    matrix = np.asarray(matrix)
    cmap = LinearSegmentedColormap.from_list(
        "complex_rank", ["#0B2D4D", "#146C78", "#78BEC7", "#EDF3F7"]
    )
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=1, vmax=18)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rank = matrix[i, j]
            if rank <= 3:
                ax.scatter(j, i, s=13, color="white", edgecolor=COLORS["ink"], linewidth=0.35, zorder=4)
            elif rank > 10:
                ax.scatter(j, i, marker="x", s=14, color="#7B8794", linewidth=0.6, zorder=4)
    top_counts = np.sum(matrix <= 10, axis=0)
    top_ax = ax.inset_axes([0.00, 1.025, 1.00, 0.13], sharex=ax)
    top_ax.bar(np.arange(matrix.shape[1]), top_counts, width=0.82, color="#AFC8DA",
               edgecolor="white", linewidth=0.35)
    top_ax.set_ylim(0, matrix.shape[0])
    top_ax.set_yticks([0, 9])
    top_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    top_ax.tick_params(axis="y", labelsize=6.4, width=0.6, length=2)
    top_ax.set_ylabel("Top-10", fontsize=6.6, fontweight="bold")
    style_axes(top_ax)
    top_ax.spines["bottom"].set_visible(False)
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color="white", linewidth=2.0)
    ax.set_xticks([0, 4, 9, 14, 19, 24])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "25"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.4)
    ax.set_xlabel("Cross-validation split")
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.036, pad=0.020)
    cbar.set_label("Rank", fontsize=7.6, fontweight="bold")
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=7.0, width=0.7, length=2.5)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["ink"], markersize=4.5, label="Top-3"),
        Line2D([0], [0], marker="x", color="#7B8794", markersize=4.5,
               linestyle="none", label=">Top-10"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.52, 1.24), ncol=2, columnspacing=0.9,
              handletextpad=0.35, borderaxespad=0.0)
    style_axes(ax)


def panel_f_gnn_enrichment(ax: plt.Axes, df: pd.DataFrame) -> None:
    method_colors = {name: color for name, _, color in METHODS}
    region_colors = {"BLA": COLORS["supp"], "CeA": COLORS["top3"], "DMS": COLORS["top10"]}

    regions = ["BLA", "CeA", "DMS"]
    base_y = {"BLA": 2.05, "CeA": 1.05, "DMS": 0.05}
    offsets = {"Consensus": 0.23, "Fused": 0.00, "GNN-lite": -0.23}

    ax.axvspan(0, 3, color="#EFF8F5", alpha=0.50, zorder=0)
    ax.axvspan(3, 10, color="#EEF5FA", alpha=0.42, zorder=0)
    ax.axvline(25, color="#C9D2DC", linewidth=0.85, linestyle=":", zorder=1)
    ax.axvline(3, color="#B9D8CC", linewidth=0.75, linestyle="--", zorder=1)
    ax.axvline(10, color="#BDD0DF", linewidth=0.75, linestyle="--", zorder=1)
    for region in regions:
        y0 = base_y[region]
        ax.axhspan(y0 - 0.36, y0 + 0.36, color=region_colors[region], alpha=0.035, zorder=0)
        ax.hlines([y0 - 0.36, y0 + 0.36], 0, 26.8, color=COLORS["grid"],
                  linewidth=0.45, alpha=0.65, zorder=0)
        ax.text(-1.68, y0, region, ha="right", va="center",
                fontsize=8.6, fontweight="bold", color=region_colors[region], clip_on=False)
        for method, col, color in METHODS:
            vals = df[df["BrainRegion"].eq(region)][col].to_numpy(float)
            top3 = int(np.sum(vals <= 3))
            top10 = int(np.sum(vals <= 10))
            y = y0 + offsets[method]
            ax.plot([0, 25], [y, y], color="#DDE5EC", linewidth=0.55,
                    alpha=0.65, zorder=1)
            ax.plot([top3, top10], [y, y], color=color, linewidth=2.05,
                    alpha=0.62, solid_capstyle="round", zorder=3)
            ax.scatter([top10], [y], s=48, marker="o", facecolor="white",
                       edgecolor=color, linewidth=1.25, zorder=5)
            ax.scatter([top3], [y], s=50, marker="D", color=color,
                       edgecolor="white", linewidth=0.55, zorder=6)
            label_y_offset = {
                "Consensus": 0.20,
                "Fused": -0.20,
                "GNN-lite": -0.20,
            }[method]
            label_va = "bottom" if label_y_offset > 0 else "top"
            ax.text(top3, y + label_y_offset, f"{top3}/25", ha="center", va=label_va,
                    fontsize=5.8, fontweight="bold", color=color, zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.25, foreground="white")])
            ax.text(top10 + 0.44, y, f"{top10}/25", ha="left", va="center",
                    fontsize=5.9, fontweight="bold", color=color, zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.25, foreground="white")])

    legend_handles = [
        Line2D([0], [0], color=color, marker="o", markersize=3.5, linewidth=1.5, label=name)
        for name, _, color in METHODS
    ] + [
        Line2D([0], [0], color=COLORS["ink"], marker="D", linestyle="none",
               markersize=4.0, label="Top-3"),
        Line2D([0], [0], color=COLORS["ink"], marker="o", markerfacecolor="white",
               linestyle="none", markersize=4.2, label="Top-10"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.50, 1.11), ncol=5, columnspacing=0.75,
              handletextpad=0.30, borderaxespad=0.0)

    ax.set_xlim(-0.4, 27.15)
    ax.set_ylim(-0.82, 2.88)
    ax.set_yticks([])
    ax.set_xticks([0, 3, 5, 10, 15, 20, 25])
    ax.set_xlabel("Retained splits out of 25")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.82)
    style_axes(ax)


def make_individual(df: pd.DataFrame) -> None:
    panels = [
        ("panel_A_rank_ridgeline", panel_a_ridgeline, (5.3, 4.0)),
        ("panel_B_rank_shift_estimation", panel_b_estimation, (5.0, 3.4)),
        ("panel_C_retention_composition", panel_c_stacked, (5.1, 3.7)),
        ("panel_D_rank_quantile_interval", panel_d_quantile_interval, (5.0, 3.2)),
        ("panel_E_coretention_patterns", panel_e_co_retention, (5.4, 3.4)),
        ("panel_F_gnn_top3_gain", panel_f_gnn_enrichment, (4.8, 3.0)),
    ]
    for stem, drawer, size in panels:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax, df)
        save_all(fig, stem)


def make_combined(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12.6, 12.2))
    gs = GridSpec(3, 2, figure=fig, wspace=0.42, hspace=0.72)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    drawers = [
        panel_a_ridgeline,
        panel_b_estimation,
        panel_c_stacked,
        panel_d_quantile_interval,
        panel_e_co_retention,
        panel_f_gnn_enrichment,
    ]
    for ax, label, drawer in zip(axes, list("ABCDEF"), drawers):
        drawer(ax, df)
        add_panel_label(ax, label)
    save_all(fig, "figure_3_4_advanced_common_combined")


def zip_outputs() -> Path:
    zip_path = ROOT / "figure_3_4_advanced_common_300dpi_tiff.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for path in sorted(OUT.glob("*")):
            if path.suffix.lower() in {".tif", ".tiff", ".png", ".pdf", ".svg", ".csv"}:
                zf.write(path, arcname=f"{OUT.name}/{path.name}")
    return zip_path


def main() -> None:
    set_style()
    df = load_data()
    df.to_csv(OUT / "figure_3_4_fold_level_source_data.csv", index=False)
    make_individual(df)
    make_combined(df)
    zip_path = zip_outputs()
    print(f"Saved advanced common Figure 3.4 panels to {OUT.resolve()}")
    print(f"Packaged outputs: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
