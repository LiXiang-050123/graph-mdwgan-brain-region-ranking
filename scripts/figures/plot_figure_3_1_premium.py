"""
Premium Figure 3.1 for Graph-MDWGAN synthetic-data assessment.

The values follow manuscript Section 3.1:
- Augmentation-factor stability at 2x, 4x, and 8x.
- 8x distribution metrics across five seeds: MMD, JS, and SWD.
- Duplicate auditing after six-decimal rounding.
- Representative real-training vs synthetic distribution overlays.
- Cross-fold stability trajectories and real-synthetic correlation-structure concordance.

Panel letters are intentionally not embedded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from scipy.stats import gaussian_kde

from analysis_paths import CODE_ROOT, CORRECTED_SECONDARY_RESULTS, SEED_RESULTS


ROOT = CODE_ROOT
OUT = ROOT / "figure_3_1_premium"
OUT.mkdir(exist_ok=True)

SEED = 20250829
FEATURES = [
    "ACC", "AM", "BLA", "CeA", "CLA", "CM", "DMS", "GP", "IL",
    "LH", "MS", "NAco", "NAsh", "OFC", "PL", "RSC", "VP", "VTA",
]
SHOW_REGIONS = ["ACC", "VTA", "NAco", "CLA", "BLA", "DMS"]

COLORS = {
    "ink": "#102A43",
    "axis": "#202020",
    "grid": "#E3E8EF",
    "soft_grid": "#F3F6F8",
    "blue": "#2E7DAE",
    "blue_light": "#BBD7E8",
    "green": "#31A77B",
    "green_light": "#BFE7D8",
    "coral": "#E05D45",
    "cyan": "#48A9D9",
    "ochre": "#C98C45",
    "teal": "#3CB7A3",
    "purple": "#8D7AC3",
    "lavender": "#D8D0EA",
    "rose": "#CF6F70",
    "gray": "#9AA3AD",
    "real": "#153957",
    "synthetic": "#45B987",
    "validation": "#A8AFB8",
}

STABILITY_CMAP = LinearSegmentedColormap.from_list(
    "stability_cmap", ["#EEF5F8", "#BFDCE3", "#6FB5BA", "#258377", "#14524D"]
)


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
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
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


def load_seed_data(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = SEED_RESULTS / f"seed_{seed}" / "ours"
    return (
        pd.read_csv(seed_dir / "train_augmented_x8.csv"),
        pd.read_csv(seed_dir / "test_original.csv"),
    )


def get_stability() -> pd.DataFrame:
    stability = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "stability_mean_sd.csv")
    return stability[
        (stability["method"] == "Ours")
        & (stability["fair"] == "raw")
        & (stability["scheme"] == "fusion")
        & (stability["k"].isin([1, 3, 5, 10]))
    ].copy()


def draw_stability_premium(ax: plt.Axes) -> None:
    data = get_stability()
    factors = [2, 4, 8]
    x = np.arange(len(factors), dtype=float)
    series = [
        (10, "Top-10", COLORS["cyan"], "D"),
        (5, "Top-5", COLORS["coral"], "^"),
        (3, "Top-3", COLORS["green"], "s"),
        (1, "Top-1", COLORS["blue"], "o"),
    ]

    ax.axvspan(1.77, 2.23, color=COLORS["green"], alpha=0.055, zorder=0)
    for k, label, color, marker in series:
        subset = data[data["k"] == k].sort_values("factor")
        y = subset["jaccard_mean"].to_numpy(dtype=float)
        sd = subset["jaccard_sd"].to_numpy(dtype=float)
        if k == 3:
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.17, linewidth=0, zorder=1)
            ax.plot(x, y, color=color, linewidth=2.4, zorder=4)
            ax.scatter(x, y, s=58, marker=marker, color=color, edgecolor="white", linewidth=0.8, zorder=5)
            for xi, yi in zip(x, y):
                ax.text(xi, yi + 0.027, f"{yi:.3f}", ha="center", va="bottom",
                        fontsize=8.1, fontweight="bold", color=color)
        else:
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.075, linewidth=0, zorder=1)
            ax.plot(x, y, color=color, linewidth=1.3, alpha=0.88, zorder=2)
            ax.scatter(x, y, s=34, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        for xi, yi in zip(x, y):
            if k != 3:
                label_x = xi
                label_y = yi + 0.006 if k == 1 and xi == 1 else yi + 0.018
                ax.text(label_x, label_y, f"{yi:.3f}", ha="center", va="bottom",
                        fontsize=7.2, fontweight="bold", color=color)
    legend_specs = [
        ("Top-1", COLORS["blue"], "o"),
        ("Top-3", COLORS["green"], "s"),
        ("Top-5", COLORS["coral"], "^"),
        ("Top-10", COLORS["cyan"], "D"),
    ]
    handles = [
        plt.Line2D([0], [0], color=color, marker=marker, linewidth=1.5,
                   markersize=5.0, markerfacecolor=color, markeredgecolor="white",
                   markeredgewidth=0.55, label=label)
        for label, color, marker in legend_specs
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              ncol=4, frameon=False, handlelength=1.5, columnspacing=1.25,
              handletextpad=0.35, borderaxespad=0.0)

    ax.text(2.0, 0.185, "8\u00d7", ha="center", va="center", fontsize=9.0,
            fontweight="bold", color=COLORS["green"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\u00d7" for f in factors])
    ax.set_xlabel("Augmentation factor")
    ax.set_ylabel("Mean Jaccard index")
    ax.set_xlim(-0.15, 2.34)
    ax.set_ylim(0.15, 0.76)
    ax.set_yticks(np.arange(0.2, 0.8, 0.1))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    style_axes(ax)


def draw_quality_metric_forest(ax: plt.Axes) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()

    raw = pd.read_csv(SEED_RESULTS / "all_seeds_long.csv")
    raw = raw[raw["factor"].eq(8)].copy()
    method_order = ["Ours", "SMOTE(no v10k)", "ROS(no v10k)"]
    method_labels = {
        "Ours": "Graph\nMDWGAN",
        "SMOTE(no v10k)": "SMOTE",
        "ROS(no v10k)": "ROS",
    }
    method_colors = {
        "Ours": COLORS["teal"],
        "SMOTE(no v10k)": COLORS["ochre"],
        "ROS(no v10k)": COLORS["gray"],
    }
    metrics = [
        ("mmd", "MMD", "Distance", 4),
        ("js", "JS divergence", "Divergence", 3),
        ("swd", "Sliced Wasserstein", "Distance", 3),
    ]
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.44)
    rng = np.random.default_rng(20260724)

    for idx, (col, label, ylabel, decimals) in enumerate(metrics):
        subax = fig.add_subplot(subgs[0, idx])
        x = np.arange(len(method_order), dtype=float)
        ymin = np.inf
        ymax = -np.inf
        for xi, method in zip(x, method_order):
            values = raw[raw["method"].eq(method)][col].astype(float).to_numpy()
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            color = method_colors[method]
            jitter = rng.normal(0, 0.055, len(values))
            subax.scatter(
                np.full(len(values), xi) + jitter,
                values,
                s=21,
                color=color,
                edgecolor="white",
                linewidth=0.50,
                alpha=0.90,
                zorder=4,
            )
            subax.errorbar(
                xi,
                mean,
                yerr=sd,
                fmt="o",
                color=COLORS["axis"],
                ecolor=COLORS["axis"],
                markerfacecolor="white",
                markeredgecolor=COLORS["axis"],
                markeredgewidth=0.85,
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
                markersize=4.7,
                zorder=5,
            )
            subax.scatter([xi], [mean], s=22, color=color, edgecolor="white",
                          linewidth=0.45, zorder=6)
            ymin = min(ymin, float(values.min()), mean - sd)
            ymax = max(ymax, float(values.max()), mean + sd)
        pad = (ymax - ymin) * 0.34 if ymax > ymin else max(ymax * 0.16, 0.001)
        if col == "mmd":
            subax.set_ylim(max(0, ymin - pad * 0.48), ymax + pad * 1.10)
            subax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
        elif col == "js":
            subax.set_ylim(max(0, ymin - pad * 0.42), ymax + pad * 0.95)
            subax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        else:
            subax.set_ylim(max(0, ymin - pad * 0.50), ymax + pad * 0.95)
            subax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ylo, yhi = subax.get_ylim()
        yrange = yhi - ylo
        for xi, method in zip(x, method_order):
            values = raw[raw["method"].eq(method)][col].astype(float).to_numpy()
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            text_y = min(mean + sd + yrange * 0.075, yhi - yrange * 0.065)
            subax.text(
                xi,
                text_y,
                f"{mean:.{decimals}f}\n\u00b1 {sd:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=5.7,
                fontweight="bold",
                color=COLORS["ink"],
                linespacing=0.82,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
                clip_on=False,
            )
        subax.text(
            0.5,
            1.04,
            label,
            transform=subax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=COLORS["ink"],
        )
        subax.set_xticks(x)
        subax.set_xticklabels([method_labels[m] for m in method_order])
        subax.set_ylabel(ylabel if idx == 0 else "")
        subax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.76)
        style_axes(subax)
        subax.tick_params(axis="x", labelsize=6.7, pad=2.0)
        subax.tick_params(axis="y", labelsize=6.8, pad=2.0)
        subax.set_xlim(-0.44, len(method_order) - 0.56)


def draw_duplicate_audit_tiles(ax: plt.Axes) -> None:
    summary = pd.read_csv(SEED_RESULTS / "summary_mean_sd_x8.csv")
    order = ["Ours", "SMOTE(no v10k)", "ROS(no v10k)"]
    labels = {"Ours": "Graph-MDWGAN", "SMOTE(no v10k)": "SMOTE", "ROS(no v10k)": "ROS"}
    rows = []
    for method in order:
        row = summary[summary["method"].eq(method)].iloc[0]
        rows.append((labels[method], float(row["dup_to_real_mean"]), float(row["dup_within_mean"])))

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    col_x = [0.52, 0.82]
    row_y = [0.73, 0.51, 0.29]
    ax.text(col_x[0], 0.91, "Synthetic vs\nreal training", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=COLORS["ink"], linespacing=0.92)
    ax.text(col_x[1], 0.91, "Within\nsynthetic samples", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=COLORS["ink"], linespacing=0.92)
    ax.plot([0.37, 0.93], [0.82, 0.82], color=COLORS["grid"], linewidth=0.85)
    for y in row_y:
        ax.plot([0.37, 0.93], [y, y], color=COLORS["grid"], linewidth=0.45, alpha=0.9, zorder=0)
    for i, (method, real, within) in enumerate(rows):
        y = row_y[i]
        is_graph = method == "Graph-MDWGAN"
        ax.text(0.05, y, method, ha="left", va="center", fontsize=8.6,
                fontweight="bold", color=COLORS["ink"] if is_graph else COLORS["gray"])
        ax.plot([0.35, 0.93], [y, y], color=COLORS["grid"], linewidth=0.55, zorder=0)
        for j, val in enumerate([real, within]):
            color = COLORS["green"] if val < 0.001 else COLORS["rose"]
            size = 360 if val < 0.001 else 820
            ax.scatter([col_x[j]], [y], s=size, facecolor=color,
                       edgecolor=color, linewidth=1.0,
                       alpha=0.16 if val < 0.001 else 0.70, zorder=2)
            if val < 0.001:
                ax.scatter([col_x[j]], [y], s=70, facecolor="white",
                           edgecolor=color, linewidth=1.15, zorder=3)
            text = "0%" if val < 0.001 else f"{val * 100:.1f}%"
            ax.text(col_x[j], y, text, ha="center", va="center", fontsize=8.6,
                    fontweight="bold", color=COLORS["ink"] if val < 0.001 else "white", zorder=4)
    ax.add_patch(FancyBboxPatch((0.36, 0.62), 0.58, 0.22,
                                boxstyle="round,pad=0.012,rounding_size=0.018",
                                facecolor=COLORS["green"], edgecolor="none",
                                alpha=0.055, zorder=-1))


def half_violin(ax: plt.Axes, values: np.ndarray, center: float, side: str, color: str,
                width: float = 0.28) -> None:
    values = np.asarray(values, dtype=float)
    if len(np.unique(np.round(values, 8))) < 3:
        return
    grid = np.linspace(values.min(), values.max(), 120)
    density = gaussian_kde(values)(grid)
    density = density / density.max() * width
    if side == "left":
        ax.fill_betweenx(grid, center - density, center, color=color, alpha=0.38, linewidth=0)
        ax.plot(center - density, grid, color=color, linewidth=0.8, alpha=0.88)
    else:
        ax.fill_betweenx(grid, center, center + density, color=color, alpha=0.38, linewidth=0)
        ax.plot(center + density, grid, color=color, linewidth=0.8, alpha=0.88)


def draw_distribution_raincloud(ax: plt.Axes) -> None:
    train, _ = load_seed_data()
    real = train[train["Origin"] == "RealTrain"]
    synth = train[train["Origin"] == "Synthetic"]
    rng = np.random.default_rng(2028)
    positions = np.arange(len(SHOW_REGIONS))
    for pos, region in zip(positions, SHOW_REGIONS):
        rvals = np.log1p(real[region].astype(float).to_numpy())
        svals = np.log1p(synth[region].astype(float).to_numpy())
        half_violin(ax, rvals, pos - 0.08, "left", COLORS["real"], width=0.22)
        half_violin(ax, svals, pos + 0.08, "right", COLORS["synthetic"], width=0.22)
        ax.boxplot([rvals], positions=[pos - 0.10], widths=0.10, patch_artist=True,
                   boxprops={"facecolor": COLORS["real"], "alpha": 0.23, "edgecolor": COLORS["real"], "linewidth": 0.7},
                   medianprops={"color": COLORS["real"], "linewidth": 1.0},
                   whiskerprops={"color": COLORS["real"], "linewidth": 0.7},
                   capprops={"color": COLORS["real"], "linewidth": 0.7},
                   flierprops={"markersize": 0})
        ax.boxplot([svals], positions=[pos + 0.10], widths=0.10, patch_artist=True,
                   boxprops={"facecolor": COLORS["synthetic"], "alpha": 0.23, "edgecolor": COLORS["synthetic"], "linewidth": 0.7},
                   medianprops={"color": COLORS["synthetic"], "linewidth": 1.0},
                   whiskerprops={"color": COLORS["synthetic"], "linewidth": 0.7},
                   capprops={"color": COLORS["synthetic"], "linewidth": 0.7},
                   flierprops={"markersize": 0})
        ax.scatter(np.full(len(rvals), pos - 0.19) + rng.normal(0, 0.018, len(rvals)), rvals,
                   s=12, color=COLORS["real"], alpha=0.78, edgecolor="none", zorder=4)
        ax.scatter(np.full(len(svals), pos + 0.19) + rng.normal(0, 0.018, len(svals)), svals,
                   s=8, color=COLORS["synthetic"], alpha=0.68, edgecolor="none", zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(SHOW_REGIONS)
    ax.set_ylabel("log(1 + c-Fos density)")
    ax.set_ylim(1.45, 5.55)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.8)
    handles = [
        plt.Line2D([0], [0], color=COLORS["real"], lw=4, alpha=0.65, label="Real training"),
        plt.Line2D([0], [0], color=COLORS["synthetic"], lw=4, alpha=0.65, label="Synthetic"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.52, 1.02),
              ncol=2, handlelength=1.4)
    style_axes(ax)


def covariance_ellipse(points: np.ndarray, ax: plt.Axes, color: str, scale: float = 2.0) -> None:
    if len(points) < 3:
        return
    cov = np.cov(points, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * scale * np.sqrt(vals)
    mean = points.mean(axis=0)
    ax.add_patch(Ellipse(mean, width, height, angle=angle, facecolor=color,
                         edgecolor=color, alpha=0.12, linewidth=1.0))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, facecolor="none",
                         edgecolor=color, alpha=0.92, linewidth=1.0))


def draw_correlation_concordance(ax: plt.Axes) -> None:
    train, _ = load_seed_data()
    real = train[train["Origin"] == "RealTrain"].copy()
    synth = train[train["Origin"] == "Synthetic"].copy()
    rows = []
    show_set = set(SHOW_REGIONS)
    for group in ["Saline", "METH"]:
        real_group = real[real["Group"] == group]
        synth_group = synth[synth["Group"] == group]
        real_corr = np.corrcoef(
            np.log1p(real_group[FEATURES].astype(float).to_numpy()), rowvar=False
        )
        synth_corr = np.corrcoef(
            np.log1p(synth_group[FEATURES].astype(float).to_numpy()), rowvar=False
        )
        for i, a in enumerate(FEATURES):
            for j, b in enumerate(FEATURES[:i]):
                rows.append({
                    "group": group,
                    "pair": f"{a}-{b}",
                    "real": real_corr[i, j],
                    "synthetic": synth_corr[i, j],
                    "candidate_pair": a in show_set and b in show_set,
                })
    corr_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    spec = ax.get_subplotspec()
    fig = ax.figure
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.18)
    pair_labels = {
        frozenset(("ACC", "VTA")): "ACC-VTA",
        frozenset(("CLA", "NAco")): "CLA-NAco",
        frozenset(("BLA", "DMS")): "BLA-DMS",
    }
    all_color = "#C7D1DB"
    all_mean_color = "#7D91A2"
    candidate_color = "#72C9A8"
    candidate_mean_color = "#218C68"

    for panel_index, group in enumerate(["Saline", "METH"]):
        subax = fig.add_subplot(subgs[0, panel_index])
        group_df = corr_df[corr_df["group"] == group].copy()
        structure_r = float(np.corrcoef(group_df["real"], group_df["synthetic"])[0, 1])
        candidate_df = group_df[group_df["candidate_pair"]]

        for row in group_df.itertuples(index=False):
            color = candidate_color if row.candidate_pair else all_color
            alpha = 0.72 if row.candidate_pair else 0.28
            linewidth = 0.75 if row.candidate_pair else 0.42
            subax.plot([0, 1], [row.real, row.synthetic], color=color,
                       alpha=alpha, linewidth=linewidth, zorder=1)

        all_means = [group_df["real"].mean(), group_df["synthetic"].mean()]
        candidate_means = [candidate_df["real"].mean(), candidate_df["synthetic"].mean()]
        subax.plot([0, 1], all_means, color=all_mean_color, linewidth=1.6,
                   marker="o", markersize=4.2, markeredgecolor="white",
                   markeredgewidth=0.45, zorder=4, label="All-pair mean")
        subax.plot([0, 1], candidate_means, color=candidate_mean_color, linewidth=2.1,
                   marker="o", markersize=4.8, markeredgecolor="white",
                   markeredgewidth=0.50, zorder=5, label="Candidate mean")

        selected = []
        for row in candidate_df.itertuples(index=False):
            left, right = row.pair.split("-", 1)
            label = pair_labels.get(frozenset((left, right)))
            if label:
                selected.append((row.synthetic, label))
                subax.scatter([1], [row.synthetic], s=24, color=candidate_mean_color,
                              edgecolor="white", linewidth=0.45, zorder=6)
        selected.sort()
        label_positions = []
        for value, label in selected:
            label_y = value
            if label_positions and label_y - label_positions[-1] < 0.065:
                label_y = label_positions[-1] + 0.065
            label_positions.append(label_y)
            subax.annotate(label, xy=(1, value), xytext=(1.04, label_y),
                           ha="left", va="center", fontsize=6.7,
                           fontweight="bold", color=COLORS["ink"],
                           arrowprops={"arrowstyle": "-", "lw": 0.55,
                                       "color": candidate_mean_color},
                           clip_on=False)

        subax.set_title(f"{group}  r = {structure_r:.2f}", fontsize=8.2,
                        fontweight="bold", color=COLORS["ink"], pad=6)
        subax.set_xticks([0, 1])
        subax.set_xticklabels(["Real training", "Synthetic"])
        subax.set_xlim(-0.08, 1.25)
        subax.set_ylim(-0.35, 1.10)
        subax.grid(axis="y", color=COLORS["grid"], linewidth=0.48, alpha=0.76)
        style_axes(subax)
        if panel_index == 0:
            subax.set_ylabel("Pairwise correlation")
            handles, labels = subax.get_legend_handles_labels()
            order = [1, 0]
            subax.legend([handles[i] for i in order], [labels[i] for i in order],
                         frameon=False, loc="upper left", handletextpad=0.35,
                         borderaxespad=0.2)
        else:
            subax.tick_params(labelleft=False)
            subax.spines["left"].set_visible(False)


def draw_paired_stability(ax: plt.Axes) -> None:
    data = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "stability_long.csv")
    top3 = data[
        (data["method"] == "Ours")
        & (data["fair"] == "raw")
        & (data["scheme"] == "fusion")
        & (data["k"] == 3)
    ].copy()
    paired = top3.pivot_table(index="fold", columns="factor", values="jaccard_mean", aggfunc="mean")
    paired = paired[[2, 4, 8]].dropna()
    factors = [2, 4, 8]
    x = np.arange(len(factors))
    colors = {2: COLORS["lavender"], 4: COLORS["ochre"], 8: COLORS["green"]}

    for _, row in paired.iterrows():
        ax.plot(x, row[factors].to_numpy(dtype=float), color="#AEB8C3", linewidth=0.70,
                alpha=0.42, zorder=1)
    for i, factor in enumerate(factors):
        vals = paired[factor].to_numpy(dtype=float)
        jitter = np.random.default_rng(9300 + factor).normal(0, 0.035, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, color=colors[factor],
                   edgecolor="white", linewidth=0.35, alpha=0.86, zorder=3)
        mean = vals.mean()
        sd = vals.std(ddof=1)
        ax.errorbar(i, mean, yerr=sd, fmt="o", markersize=6.2,
                    color=COLORS["axis"], ecolor=COLORS["axis"],
                    markerfacecolor="white", markeredgewidth=1.0,
                    elinewidth=1.15, capsize=4.0, zorder=5)
        ax.scatter([i], [mean], s=38, color=colors[factor], edgecolor="white",
                   linewidth=0.65, zorder=6)
        ax.text(i + 0.08, mean + 0.020, f"{mean:.3f}", ha="left", va="bottom",
                fontsize=7.7, fontweight="bold", color=COLORS["ink"],
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.76),
                zorder=7)
    ax.plot(x, paired[factors].mean(axis=0).to_numpy(dtype=float),
            color=COLORS["green"], linewidth=2.0, alpha=0.90, zorder=4)
    ax.fill_between(
        x,
        paired[factors].quantile(0.25).to_numpy(dtype=float),
        paired[factors].quantile(0.75).to_numpy(dtype=float),
        color=COLORS["green"],
        alpha=0.10,
        linewidth=0,
        zorder=0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\u00d7" for f in factors])
    ax.set_ylabel("Top-3 Jaccard index")
    ax.set_xlabel("Augmentation factor")
    ax.set_ylim(0.24, 0.62)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.80)
    ax.grid(axis="x", color=COLORS["soft_grid"], linewidth=0.45, alpha=0.65)
    style_axes(ax)


def make_figures() -> None:
    panel_specs = [
        ("premium_panel_A_stability", draw_stability_premium, (5.8, 3.25)),
        ("premium_panel_B_quality_forest", draw_quality_metric_forest, (7.3, 3.15)),
        ("premium_panel_C_duplicate_tiles", draw_duplicate_audit_tiles, (4.7, 2.75)),
        ("premium_panel_D_distribution_raincloud", draw_distribution_raincloud, (6.2, 3.25)),
        ("premium_panel_E_correlation_concordance", draw_correlation_concordance, (4.95, 3.85)),
        ("premium_panel_F_paired_stability", draw_paired_stability, (4.9, 3.05)),
    ]
    for stem, drawer, size in panel_specs:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax)
        save_all(fig, stem)

    fig = plt.figure(figsize=(13.8, 10.15))
    gs = GridSpec(
        3, 12, figure=fig,
        height_ratios=[1.03, 1.03, 0.84],
        width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        wspace=0.86,
        hspace=0.50,
    )
    axes = {
        "A": fig.add_subplot(gs[0, 0:6]),
        "B": fig.add_subplot(gs[0, 6:12]),
        "D": fig.add_subplot(gs[1, 0:7]),
        "E": fig.add_subplot(gs[1, 7:12]),
        "C": fig.add_subplot(gs[2, 0:5]),
        "F": fig.add_subplot(gs[2, 5:12]),
    }
    draw_stability_premium(axes["A"])
    draw_quality_metric_forest(axes["B"])
    draw_duplicate_audit_tiles(axes["C"])
    draw_paired_stability(axes["F"])
    draw_correlation_concordance(axes["E"])
    draw_distribution_raincloud(axes["D"])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_3_1_premium_combined.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    make_figures()
    print(f"Saved premium Figure 3.1 to {OUT.resolve()}")


if __name__ == "__main__":
    main()
