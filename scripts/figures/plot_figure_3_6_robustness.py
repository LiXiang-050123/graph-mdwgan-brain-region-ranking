"""
Premium Figure 3.6 panels for baseline and sensitivity analyses.

The panels use the 8x, raw, fused top-3, logistic-regression protocol for the
main baseline comparison, then draw classifier, preprocessing, duplication, and
low-data sensitivity views from the full result tables.
Panel letters and large titles are intentionally not embedded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle, Wedge
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde

from analysis_paths import (
    CODE_ROOT,
    CORRECTED_SECONDARY_RESULTS,
    LOW_DATA_RESULTS,
)


ROOT = CODE_ROOT
OUT = ROOT / "figure_3_6_robustness"
OUT.mkdir(exist_ok=True)

N_FOLDS = 25
METHOD_ORDER = ["Graph-MDWGAN", "RealOnly", "ROS", "SMOTE"]
METHOD_KEY = {
    "Ours": "Graph-MDWGAN",
    "RealOnly": "RealOnly",
    "ROS": "ROS",
    "SMOTE": "SMOTE",
}
REV_METHOD_KEY = {v: k for k, v in METHOD_KEY.items()}

COLORS = {
    "ink": "#102A43",
    "axis": "#202020",
    "grid": "#E5EAF0",
    "pale": "#F7F9FB",
    "graph": "#2FAE8F",
    "real": "#355C7D",
    "ros": "#C9784B",
    "smote": "#7D6EB2",
    "gold": "#D8A441",
    "rose": "#D66761",
    "sky": "#5AA9D6",
    "soft_green": "#E9F6F1",
    "soft_blue": "#ECF4FA",
    "soft_orange": "#FBF0E8",
    "soft_purple": "#F1EEF8",
}

METHOD_COLORS = {
    "Graph-MDWGAN": COLORS["graph"],
    "RealOnly": COLORS["real"],
    "ROS": COLORS["ros"],
    "SMOTE": COLORS["smote"],
}

METRICS = [
    ("AUC", "auc_mean", "auc_sd"),
    ("AUPRC", "auprc_mean", "auprc_sd"),
    ("Balanced accuracy", "balacc_mean", "balacc_sd"),
    ("MCC", "mcc_mean", "mcc_sd"),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_mean = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "perf_mean_sd.csv")
    final_long = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "perf_long.csv")
    baseline = final_mean[
        (final_mean["factor"] == 8)
        & (final_mean["fair"] == "raw")
        & (final_mean["scheme"] == "fusion")
        & (final_mean["k"] == 3)
        & (final_mean["clf"] == "logreg")
    ].copy()
    low_mean = pd.read_csv(LOW_DATA_RESULTS / "perf_lowdata_mean_sd.csv")
    low_long = pd.read_csv(LOW_DATA_RESULTS / "perf_lowdata_long.csv")

    for frame in (baseline, final_mean, final_long, low_mean, low_long):
        if "method" in frame.columns:
            frame["label"] = frame["method"].map(METHOD_KEY).fillna(frame["method"])
    return baseline, final_mean, final_long, low_mean, low_long


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


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(axis="both", colors="black", labelcolor="black", pad=2)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Arial")
        tick.set_fontweight("bold")


def save_all(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", dpi=600, bbox_inches="tight")
    plt.close(fig)


def ci95(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if values.size == 1:
        return mean, mean, mean
    half = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    return mean, mean - half, mean + half


def baseline_ordered(baseline: pd.DataFrame) -> pd.DataFrame:
    frame = baseline.copy()
    frame["label"] = frame["label"].replace({"Ours": "Graph-MDWGAN"})
    frame["label"] = pd.Categorical(frame["label"], METHOD_ORDER, ordered=True)
    return frame.sort_values("label")


def panel_a_metric_forest(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline)
    y_base = np.arange(len(METHOD_ORDER))[::-1]
    offsets = np.array([0.30, 0.10, -0.10, -0.30])
    metric_colors = [COLORS["graph"], COLORS["sky"], COLORS["gold"], COLORS["rose"]]
    for y, label in zip(y_base, METHOD_ORDER):
        ax.plot([0.63, 1.055], [y, y], color=COLORS["grid"], linewidth=0.6, zorder=0)
        row = df[df["label"].eq(label)].iloc[0]
        ax.text(-0.135, y, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.8,
                fontweight="bold", color=METHOD_COLORS[label], clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 0.85})
        for j, (metric, mean_col, sd_col) in enumerate(METRICS):
            yj = y + offsets[j]
            mean = float(row[mean_col])
            sd = float(row[sd_col])
            lo = max(0, mean - sd)
            hi = min(1.02, mean + sd)
            ax.plot([lo, hi], [yj, yj], color=metric_colors[j], linewidth=1.35,
                    solid_capstyle="round", alpha=0.90)
            ax.scatter(mean, yj, s=38, color=metric_colors[j], edgecolor="white",
                       linewidth=0.65, zorder=3)
            ax.text(1.028, yj, f"{mean:.3f}", ha="left", va="center",
                    fontsize=6.3, fontweight="bold", color=metric_colors[j])
    ax.set_xlim(0.63, 1.085)
    ax.set_ylim(-0.75, len(METHOD_ORDER) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("Cross-validated metric value")
    ax.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    handles = [
        Line2D([0], [0], marker="o", color=color, label=metric, markersize=5.2,
               markerfacecolor=color, markeredgecolor="white", linewidth=1.2)
        for (metric, _, _), color in zip(METRICS, metric_colors)
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=4, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.28)
    style_axes(ax)


def panel_b_paired_auc_delta(ax: plt.Axes, long_df: pd.DataFrame) -> None:
    sub = long_df[
        (long_df["factor"].eq(8)) & (long_df["fair"].eq("raw")) &
        (long_df["scheme"].eq("fusion")) & (long_df["k"].eq(3)) &
        (long_df["clf"].eq("logreg"))
    ].copy()
    piv = sub.pivot_table(index="fold", columns="label", values="auc")
    comparisons = ["RealOnly", "ROS", "SMOTE"]
    y_positions = np.arange(len(comparisons))[::-1]
    rng = np.random.default_rng(3601)
    for y, comp in zip(y_positions, comparisons):
        delta = (piv["Graph-MDWGAN"] - piv[comp]).dropna().to_numpy(dtype=float)
        jitter = rng.normal(0, 0.035, size=len(delta))
        ax.scatter(delta, np.full_like(delta, y) + jitter, s=20,
                   color=METHOD_COLORS[comp], alpha=0.46, edgecolor="white",
                   linewidth=0.35, zorder=2)
        mean, lo, hi = ci95(delta)
        ax.plot([lo, hi], [y, y], color=COLORS["ink"], linewidth=1.25, zorder=3)
        ax.scatter(mean, y, marker="D", s=58, color=METHOD_COLORS["Graph-MDWGAN"],
                   edgecolor="white", linewidth=0.75, zorder=4)
        mean_label = "+0.000" if abs(mean) < 0.0005 else f"{mean:+.3f}"
        ax.text(0.205, y, mean_label, ha="left", va="center",
                fontsize=7.2, fontweight="bold", color=COLORS["ink"])
    ax.axvline(0, color=COLORS["axis"], linewidth=0.8)
    ax.axvspan(-0.22, 0, color=COLORS["soft_orange"], alpha=0.55, zorder=0)
    ax.axvspan(0, 0.22, color=COLORS["soft_green"], alpha=0.55, zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"vs {x}" for x in comparisons])
    ax.set_xlim(-0.22, 0.245)
    ax.set_xlabel("Fold-wise AUC difference")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    style_axes(ax)


def panel_c_classifier_sensitivity(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    x = np.arange(len(clf_order))
    for label in METHOD_ORDER:
        row = sub[sub["label"].eq(label)].set_index("clf").loc[clf_order]
        vals = row["auc_mean"].to_numpy(dtype=float)
        ax.plot(x, vals, color=METHOD_COLORS[label], linewidth=1.35,
                alpha=0.92, zorder=2)
        ax.scatter(x, vals, s=62, color=METHOD_COLORS[label], edgecolor="white",
                   linewidth=0.75, zorder=3)
        ax.text(x[-1] + 0.07, vals[-1], label, ha="left", va="center",
                fontsize=6.6, fontweight="bold", color=METHOD_COLORS[label])
    ax.set_xticks(x)
    ax.set_xticklabels(["Logistic", "Linear SVM", "Random forest"])
    ax.set_ylabel("Mean AUC")
    ax.set_xlim(-0.18, len(clf_order) - 0.10)
    ax.set_ylim(0.88, 1.005)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    style_axes(ax)


def panel_d_preprocessing_range(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["label"].eq("Graph-MDWGAN")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3)) &
        (mean_df["clf"].eq("logreg"))
    ].copy()
    fair_order = ["raw", "dedup", "fixunique"]
    x = np.arange(len(METRICS))
    fair_offsets = {"raw": -0.16, "dedup": 0.0, "fixunique": 0.16}
    fair_colors = {"raw": COLORS["graph"], "dedup": COLORS["sky"], "fixunique": COLORS["gold"]}
    for j, (metric, mean_col, _) in enumerate(METRICS):
        vals = sub.set_index("fair").loc[fair_order, mean_col].to_numpy(dtype=float)
        ax.plot([j, j], [vals.min(), vals.max()], color=COLORS["axis"],
                linewidth=1.1, alpha=0.65, zorder=1)
        for fair in fair_order:
            val = float(sub[sub["fair"].eq(fair)][mean_col].iloc[0])
            ax.scatter(j + fair_offsets[fair], val, s=58, marker="o",
                       color=fair_colors[fair], edgecolor="white",
                       linewidth=0.70, zorder=3)
        ax.text(j, vals.max() + 0.018, f"range {vals.max() - vals.min():.3f}",
                ha="center", va="bottom", fontsize=6.4, fontweight="bold",
                color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(["AUC", "AUPRC", "Bal. acc.", "MCC"])
    ax.set_ylabel("Graph-MDWGAN metric value")
    ax.set_ylim(0.80, 1.02)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    handles = [Line2D([0], [0], marker="o", color="none", label=fair,
                      markerfacecolor=fair_colors[fair], markeredgecolor="white",
                      markersize=6) for fair in fair_order]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.52, 1.02), ncol=3, columnspacing=0.8,
              handletextpad=0.28)
    style_axes(ax)


def panel_e_duplicate_audit(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline)
    ax.axvspan(-0.03, 0.08, color=COLORS["soft_green"], alpha=0.70, zorder=0)
    ax.axhspan(-0.03, 0.08, color=COLORS["soft_green"], alpha=0.70, zorder=0)
    display_offsets = {
        "Graph-MDWGAN": (0.020, 0.012),
        "RealOnly": (0.000, -0.020),
        "SMOTE": (-0.020, 0.018),
        "ROS": (0.000, 0.000),
    }
    label_pos = {
        "Graph-MDWGAN": (0.28, 0.115),
        "SMOTE": (0.28, 0.055),
        "RealOnly": (0.28, -0.010),
        "ROS": (0.92, 0.795),
    }
    for _, row in df.iterrows():
        label = str(row["label"])
        x = float(row["dup_to_real_mean"])
        y = float(row["dup_within_mean"])
        dx_point, dy_point = display_offsets[label]
        px, py = x + dx_point, y + dy_point
        size = 170 if label == "ROS" else 115
        ax.scatter(px, py, s=size, color=METHOD_COLORS[label], edgecolor="white",
                   linewidth=0.9, zorder=3, alpha=0.92)
        tx, ty = label_pos[label]
        ax.plot([px, tx - 0.012], [py, ty], color=METHOD_COLORS[label],
                linewidth=0.65, alpha=0.78, zorder=2)
        ax.text(tx, ty, label, ha="left" if label != "ROS" else "right",
                va="center", fontsize=7.0, fontweight="bold",
                color=METHOD_COLORS[label])
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 0.96)
    ax.set_xlabel("Synthetic-real duplicate rate")
    ax.set_ylabel("Within-synthetic duplicate rate")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 0.85])
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    style_axes(ax)


def panel_f_lowdata_delta(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["label"].eq("Graph-MDWGAN")) &
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg"))
    ].copy()
    piv = sub.pivot_table(index=["fold", "frac_real"], columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    fracs = sorted(piv["frac_real"].unique())
    means, lows, highs = [], [], []
    for frac in fracs:
        m, lo, hi = ci95(piv[piv["frac_real"].eq(frac)]["delta"].to_numpy())
        means.append(m)
        lows.append(lo)
        highs.append(hi)
    ax.axhline(0, color=COLORS["axis"], linewidth=0.8)
    ax.fill_between(fracs, lows, highs, color=COLORS["graph"], alpha=0.18, zorder=1)
    ax.plot(fracs, means, color=COLORS["graph"], linewidth=1.7, zorder=3)
    ax.scatter(fracs, means, s=58, color=COLORS["graph"], edgecolor="white",
               linewidth=0.75, zorder=4)
    for frac, mean in zip(fracs, means):
        if frac in [0.2, 1.0]:
            ax.text(frac, mean + 0.022, f"{mean:+.3f}", ha="center",
                    va="bottom", fontsize=7.0, fontweight="bold",
                    color=COLORS["graph"])
    ax.set_xlabel("Fraction of real training samples")
    ax.set_ylabel("Balanced accuracy gain")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_ylim(-0.08, 0.12)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    style_axes(ax)


def half_violin(ax: plt.Axes, values: np.ndarray, center: float, color: str,
                side: str = "left", width: float = 0.34) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, values[0]):
        ax.scatter([center], [values.mean() if values.size else np.nan],
                   s=40, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        return
    grid = np.linspace(max(0.45, values.min() - 0.05), min(1.02, values.max() + 0.05), 160)
    density = gaussian_kde(values)(grid)
    density = density / density.max() * width
    if side == "left":
        xs = center - density
    else:
        xs = center + density
    ax.fill_betweenx(grid, center, xs, color=color, alpha=0.24, linewidth=0)
    ax.plot(xs, grid, color=color, linewidth=1.05, alpha=0.90)


def panel_g_lowdata_raincloud(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["label"].eq("Graph-MDWGAN")) & (low_long["frac_real"].eq(0.2)) &
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg"))
    ].copy()
    rng = np.random.default_rng(3602)
    modes = [("real", "RealOnly", COLORS["real"], "left"), ("aug", "Augmented", COLORS["graph"], "right")]
    for i, (mode, label, color, side) in enumerate(modes):
        vals = sub[sub["train_mode"].eq(mode)]["bacc"].to_numpy(dtype=float)
        half_violin(ax, vals, i, color, side=side)
        jitter = rng.normal(0, 0.035, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=color,
                   edgecolor="white", linewidth=0.35, alpha=0.65, zorder=3)
        mean, lo, hi = ci95(vals)
        ax.plot([i - 0.12, i + 0.12], [mean, mean], color=COLORS["ink"],
                linewidth=1.15, zorder=4)
        ax.plot([i, i], [lo, hi], color=COLORS["ink"], linewidth=1.0, zorder=4)
        ax.text(i, 1.015, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=7.2, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RealOnly", "Augmented"])
    ax.set_ylabel("Balanced accuracy at 20% real data")
    ax.set_ylim(0.45, 1.06)
    ax.set_xlim(-0.65, 1.65)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    style_axes(ax)


def panel_h_winner_map(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    metrics = [("AUC", "auc_mean"), ("AUPRC", "auprc_mean"),
               ("Bal. acc.", "balacc_mean"), ("MCC", "mcc_mean")]
    cells = []
    for clf in clf_order:
        for metric, col in metrics:
            block = sub[sub["clf"].eq(clf)].copy()
            best_value = block[col].max()
            winners = block[np.isclose(block[col], best_value)]["label"].tolist()
            cells.append({"clf": clf, "metric": metric, "winner": winners[0],
                          "tie": len(winners) > 1, "value": best_value})
    x_labels = [m[0] for m in metrics]
    y_labels = ["Logistic", "Linear SVM", "Random forest"]
    for yi, clf in enumerate(clf_order[::-1]):
        for xi, metric in enumerate(x_labels):
            rec = next(c for c in cells if c["clf"] == clf and c["metric"] == metric)
            color = METHOD_COLORS[rec["winner"]]
            rect = FancyBboxPatch(
                (xi - 0.38, yi - 0.34), 0.76, 0.68,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=color, edgecolor="white", linewidth=0.9, alpha=0.82,
            )
            ax.add_patch(rect)
            text = rec["winner"].replace("Graph-MDWGAN", "Graph")
            if rec["tie"]:
                text += "*"
            ax.text(xi, yi + 0.06, text, ha="center", va="center",
                    fontsize=6.3, fontweight="bold", color="white")
            ax.text(xi, yi - 0.16, f"{rec['value']:.3f}", ha="center", va="center",
                    fontsize=5.8, fontweight="bold", color="white")
    ax.set_xlim(-0.55, len(x_labels) - 0.45)
    ax.set_ylim(-0.55, len(y_labels) - 0.45)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels[::-1])
    ax.tick_params(length=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(len(x_labels) - 0.48, -0.88, "* tied maximum", ha="right", va="center",
            fontsize=6.2, fontweight="bold", color=COLORS["ink"])


def panel_i_loss_pareto(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    markers = {"logreg": "o", "svm": "D", "rf": "s"}
    for label in METHOD_ORDER:
        block = sub[sub["label"].eq(label)]
        ax.plot(block["logloss_mean"], block["brier_mean"], color=METHOD_COLORS[label],
                linewidth=0.9, alpha=0.42, zorder=1)
        for _, row in block.iterrows():
            ax.scatter(row["logloss_mean"], row["brier_mean"],
                       s=48 + 170 * (row["auc_mean"] - 0.88),
                       marker=markers[row["clf"]], color=METHOD_COLORS[label],
                       edgecolor="white", linewidth=0.7, alpha=0.92, zorder=3)
    ax.annotate("lower is better", xy=(0.19, 0.069), xytext=(0.30, 0.128),
                arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["ink"]},
                ha="left", va="center", fontsize=6.6, fontweight="bold",
                color=COLORS["ink"])
    ax.set_xlabel("Mean log loss")
    ax.set_ylabel("Mean Brier score")
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    handles = [Line2D([0], [0], marker=markers[k], color="none",
                      markerfacecolor="#7B8794", markeredgecolor="white",
                      markersize=5.5, label={"logreg": "Logistic", "svm": "SVM", "rf": "RF"}[k])
               for k in ["logreg", "svm", "rf"]]
    ax.legend(handles=handles, frameon=False, loc="upper right",
              handletextpad=0.25, borderaxespad=0.2)
    style_axes(ax)


def pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=float)
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0, ddof=1) > 0, x.std(axis=0, ddof=1), 1)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s ** 2) / np.sum(s ** 2)
    return coords, explained[:2]


def covariance_ellipse(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if len(points) < 3:
        return
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ell = Ellipse(points.mean(axis=0), width=2.15 * np.sqrt(vals[0]),
                  height=2.15 * np.sqrt(vals[1]), angle=angle,
                  facecolor=color, edgecolor=color, alpha=0.10, linewidth=1.1, zorder=0)
    ax.add_patch(ell)


def panel_c_performance_embedding(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    marker_map = {"logreg": "o", "svm": "D", "rf": "s"}
    metric_cols = ["auc_mean", "auprc_mean", "balacc_mean", "mcc_mean"]
    sub["label"] = pd.Categorical(sub["label"], METHOD_ORDER, ordered=True)
    sub["clf"] = pd.Categorical(sub["clf"], clf_order, ordered=True)
    sub = sub.sort_values(["label", "clf"]).reset_index(drop=True)
    coords, explained = pca_2d(sub[metric_cols].to_numpy())
    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    ax.axhline(0, color=COLORS["grid"], linewidth=0.75, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.75, zorder=0)
    for label in METHOD_ORDER:
        block = sub[sub["label"].eq(label)]
        pts = block[["PC1", "PC2"]].to_numpy()
        covariance_ellipse(ax, pts, METHOD_COLORS[label])
        ax.plot(block["PC1"], block["PC2"], color=METHOD_COLORS[label],
                linewidth=0.85, alpha=0.48, zorder=1)
        for _, row in block.iterrows():
            ax.scatter(row["PC1"], row["PC2"], s=70, marker=marker_map[row["clf"]],
                       color=METHOD_COLORS[label], edgecolor="white",
                       linewidth=0.75, zorder=3)
        centroid = pts.mean(axis=0)
        ax.text(centroid[0], centroid[1] + 0.28, label.replace("Graph-MDWGAN", "Graph-MDWGAN"),
                ha="center", va="center", fontsize=6.6, fontweight="bold",
                color=METHOD_COLORS[label])
    handles = [
        Line2D([0], [0], marker=marker_map[c], color="none",
               markerfacecolor="#7B8794", markeredgecolor="white",
               markersize=5.8, label={"logreg": "Logistic", "svm": "SVM", "rf": "RF"}[c])
        for c in clf_order
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.52, 1.02), ncol=3, handletextpad=0.25,
              columnspacing=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.50)
    style_axes(ax)


def panel_d_processing_glyphs(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["label"].eq("Graph-MDWGAN")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3)) &
        (mean_df["clf"].eq("logreg"))
    ].copy()
    fair_order = ["raw", "dedup", "fixunique"]
    fair_colors = {"raw": COLORS["graph"], "dedup": COLORS["sky"], "fixunique": COLORS["gold"]}
    metric_labels = ["AUC", "AUPRC", "Bal. acc.", "MCC"]
    centers = [(0.18, 0.62), (0.40, 0.62), (0.62, 0.62), (0.84, 0.62)]
    for center, (metric_label, mean_col, _), display_label in zip(centers, METRICS, metric_labels):
        vals = sub.set_index("fair").loc[fair_order, mean_col].to_numpy(dtype=float)
        base = vals.min() - 0.015
        span = max(vals.max() - base, 0.03)
        for j, fair in enumerate(fair_order):
            value = float(sub[sub["fair"].eq(fair)][mean_col].iloc[0])
            frac = np.clip((value - base) / span, 0.05, 1.0)
            radius = 0.088 - j * 0.022
            ax.add_patch(Wedge(center, radius, 90, 90 + 360 * frac,
                               width=0.014, facecolor=fair_colors[fair],
                               edgecolor="none", alpha=0.95))
            ax.add_patch(Wedge(center, radius, 90 + 360 * frac, 450,
                               width=0.014, facecolor=COLORS["grid"],
                               edgecolor="none", alpha=0.70))
        ax.text(center[0], center[1], f"{vals.mean():.3f}", ha="center", va="center",
                fontsize=6.5, fontweight="bold", color=COLORS["ink"])
        ax.text(center[0], center[1] - 0.130, display_label, ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=COLORS["ink"])
        ax.text(center[0], center[1] + 0.135, f"range {vals.max() - vals.min():.3f}",
                ha="center", va="center", fontsize=5.9, fontweight="bold",
                color=COLORS["ink"])
    for i, fair in enumerate(fair_order):
        x0 = 0.31 + i * 0.14
        ax.plot([x0, x0 + 0.040], [0.18, 0.18], color=fair_colors[fair],
                linewidth=3.0, solid_capstyle="round")
        ax.text(x0 + 0.048, 0.18, fair, ha="left", va="center",
                fontsize=6.4, fontweight="bold", color=COLORS["ink"])
    ax.set_xlim(0.04, 0.98)
    ax.set_ylim(0.08, 0.83)
    ax.axis("off")


def panel_e_duplicate_signature(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    lanes = [("Synthetic-real", "dup_to_real_mean", 0.28), ("Within-synthetic", "dup_within_mean", 0.72)]
    for lane_label, col, y0 in lanes:
        ax.plot([0, 1.0], [y0, y0], color=COLORS["grid"], linewidth=3.2,
                solid_capstyle="round", zorder=0)
        ax.text(-0.09, y0, lane_label, ha="right", va="center",
                fontsize=7.2, fontweight="bold", color=COLORS["ink"])
        for _, row in df.iterrows():
            label = row["label"]
            value = float(row[col])
            y = y0 + {"Graph-MDWGAN": 0.050, "RealOnly": 0.016, "SMOTE": -0.016, "ROS": -0.050}[label]
            ax.plot([0, value], [y, y], color=METHOD_COLORS[label],
                    linewidth=1.5, alpha=0.75, solid_capstyle="round")
            ax.scatter(value, y, s=58, color=METHOD_COLORS[label], edgecolor="white",
                       linewidth=0.70, zorder=3)
            txt = "0.000" if abs(value) < 0.0005 else f"{value:.3f}"
            ha = "left" if value < 0.78 else "right"
            tx = value + 0.025 if value < 0.78 else value - 0.035
            ax.text(tx, y, txt, ha=ha, va="center", fontsize=6.3,
                    fontweight="bold", color=METHOD_COLORS[label])
    handles = [Line2D([0], [0], marker="o", color=METHOD_COLORS[m],
                      markerfacecolor=METHOD_COLORS[m], markeredgecolor="white",
                      markersize=5.4, label=m.replace("Graph-MDWGAN", "Graph"))
               for m in METHOD_ORDER]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=4, columnspacing=0.7,
              handletextpad=0.25)
    ax.set_xlim(-0.13, 1.08)
    ax.set_ylim(0.05, 0.94)
    ax.set_xlabel("Duplicate rate")
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    style_axes(ax)


def panel_f_lowdata_ridgelines(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["label"].eq("Graph-MDWGAN")) &
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg"))
    ].copy()
    piv = sub.pivot_table(index=["fold", "frac_real"], columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    fracs = sorted(piv["frac_real"].unique())
    x_grid = np.linspace(-0.26, 0.26, 240)
    for yi, frac in enumerate(fracs[::-1]):
        vals = piv[piv["frac_real"].eq(frac)]["delta"].to_numpy(dtype=float)
        if len(np.unique(np.round(vals, 5))) > 1:
            dens = gaussian_kde(vals)(x_grid)
            dens = dens / dens.max() * 0.34
        else:
            dens = np.zeros_like(x_grid)
        base = yi
        ax.fill_between(x_grid, base, base + dens, color=COLORS["graph"],
                        alpha=0.24 + 0.08 * (1 - frac), linewidth=0)
        ax.plot(x_grid, base + dens, color=COLORS["graph"], linewidth=1.05)
        mean, lo, hi = ci95(vals)
        ax.plot([lo, hi], [base + 0.42, base + 0.42], color=COLORS["ink"],
                linewidth=0.95)
        ax.scatter(mean, base + 0.42, marker="D", s=40,
                   color=COLORS["graph"], edgecolor="white", linewidth=0.65, zorder=3)
        ax.text(-0.285, base + 0.12, f"{int(frac * 100)}%", ha="right",
                va="center", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
        if frac in {0.2, 1.0}:
            ax.text(mean + 0.018, base + 0.42, f"{mean:+.3f}", ha="left",
                    va="center", fontsize=6.3, fontweight="bold", color=COLORS["graph"])
    ax.axvline(0, color=COLORS["axis"], linewidth=0.8)
    ax.set_xlim(-0.30, 0.30)
    ax.set_ylim(-0.10, len(fracs) - 0.35)
    ax.set_xlabel("Augmented minus RealOnly balanced accuracy")
    ax.set_yticks([])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def panel_g_lowdata_paired_lanes(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["label"].eq("Graph-MDWGAN")) & (low_long["frac_real"].eq(0.2)) &
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg"))
    ].copy()
    piv = sub.pivot_table(index="fold", columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    piv = piv.sort_values("delta").reset_index(drop=True)
    y = np.arange(len(piv))
    for i, row in piv.iterrows():
        color = COLORS["graph"] if row["delta"] >= 0 else COLORS["rose"]
        ax.plot([row["real"], row["aug"]], [i, i], color=color,
                linewidth=1.0, alpha=0.62, zorder=1)
        ax.scatter(row["real"], i, s=18, color=COLORS["real"],
                   edgecolor="white", linewidth=0.35, zorder=3, alpha=0.85)
        ax.scatter(row["aug"], i, s=22, color=COLORS["graph"],
                   edgecolor="white", linewidth=0.35, zorder=3, alpha=0.90)
    ax.axvline(piv["real"].mean(), color=COLORS["real"], linewidth=1.1, alpha=0.85)
    ax.axvline(piv["aug"].mean(), color=COLORS["graph"], linewidth=1.1, alpha=0.85)
    ax.text(piv["real"].mean(), len(piv) + 0.6, f"{piv['real'].mean():.3f}",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=COLORS["real"])
    ax.text(piv["aug"].mean(), len(piv) + 0.6, f"{piv['aug'].mean():.3f}",
            ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=COLORS["graph"])
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["real"],
               markeredgecolor="white", markersize=5.2, label="RealOnly"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["graph"],
               markeredgecolor="white", markersize=5.2, label="Augmented"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=2, handletextpad=0.25,
              columnspacing=0.9)
    ax.set_xlabel("Balanced accuracy at 20% real data")
    ax.set_ylabel("Cross-validation split")
    ax.set_xlim(0.45, 1.03)
    ax.set_ylim(-0.8, len(piv) + 1.0)
    ax.set_yticks([0, 12, 24])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.75)
    style_axes(ax)


def panel_h_dominance_orbits(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    metrics = [("AUC", "auc_mean"), ("AUPRC", "auprc_mean"),
               ("Bal. acc.", "balacc_mean"), ("MCC", "mcc_mean")]
    cells = []
    for clf in clf_order:
        for metric, col in metrics:
            block = sub[sub["clf"].eq(clf)].copy()
            best_value = block[col].max()
            winners = set(block[np.isclose(block[col], best_value)]["label"].tolist())
            cells.append((clf, metric, winners))
    centers = [(0.18, 0.55), (0.40, 0.55), (0.62, 0.55), (0.84, 0.55)]
    theta = np.linspace(90, 450, len(cells), endpoint=False)
    for center, label in zip(centers, METHOD_ORDER):
        cx, cy = center
        ax.add_patch(Circle(center, 0.057, facecolor=METHOD_COLORS[label],
                            edgecolor="white", linewidth=1.0, alpha=0.92))
        ax.text(cx, cy, label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real"),
                ha="center", va="center", fontsize=6.1, fontweight="bold", color="white")
        wins = 0
        for angle, (_, _, winners) in zip(theta, cells):
            rad = np.deg2rad(angle)
            px = cx + 0.095 * np.cos(rad)
            py = cy + 0.095 * np.sin(rad)
            won = label in winners
            wins += int(won)
            ax.scatter(px, py, s=28 if won else 18,
                       color=METHOD_COLORS[label] if won else COLORS["grid"],
                       edgecolor="white", linewidth=0.55, alpha=0.95 if won else 0.70)
        ax.text(cx, cy - 0.155, f"{wins}/12", ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=METHOD_COLORS[label])
    ax.text(0.50, 0.16, "12 metric-classifier cells", ha="center", va="center",
            fontsize=6.4, fontweight="bold", color=COLORS["ink"])
    ax.set_xlim(0.04, 0.98)
    ax.set_ylim(0.06, 0.90)
    ax.axis("off")


def panel_i_calibration_glyphs(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    x_map = {clf: i for i, clf in enumerate(clf_order)}
    y_map = {label: i for i, label in enumerate(METHOD_ORDER[::-1])}
    ll = sub["logloss_mean"].to_numpy(dtype=float)
    br = sub["brier_mean"].to_numpy(dtype=float)
    ll_norm = (ll - ll.min()) / max(ll.max() - ll.min(), 1e-9)
    br_norm = (br - br.min()) / max(br.max() - br.min(), 1e-9)
    sub = sub.copy()
    sub["ll_norm"] = ll_norm
    sub["br_norm"] = br_norm
    for _, row in sub.iterrows():
        x = x_map[row["clf"]]
        y = y_map[row["label"]]
        color = METHOD_COLORS[row["label"]]
        outer = 0.085 + 0.075 * row["ll_norm"]
        inner = 0.026 + 0.052 * row["br_norm"]
        ax.add_patch(Circle((x, y), outer, facecolor="none",
                            edgecolor=color, linewidth=1.25, alpha=0.86))
        ax.add_patch(Circle((x, y), inner, facecolor=color,
                            edgecolor="white", linewidth=0.60, alpha=0.78))
        ax.text(x, y - 0.25, f"{row['logloss_mean']:.2f}/{row['brier_mean']:.2f}",
                ha="center", va="center", fontsize=5.5, fontweight="bold",
                color=COLORS["ink"])
    ax.set_xticks(range(len(clf_order)))
    ax.set_xticklabels(["Logistic", "SVM", "RF"])
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([m.replace("Graph-MDWGAN", "Graph") for m in METHOD_ORDER[::-1]])
    ax.set_xlim(-0.55, len(clf_order) - 0.45)
    ax.set_ylim(-0.55, len(METHOD_ORDER) - 0.45)
    ax.tick_params(length=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(2.48, 3.35, "outer: log loss\ninner: Brier", ha="right", va="top",
            fontsize=6.2, fontweight="bold", color=COLORS["ink"])


def panel_c_performance_embedding(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[mean_df["factor"].eq(8)].copy()
    metric_cols = ["auc_mean", "auprc_mean", "balacc_mean", "mcc_mean"]
    features = sub[metric_cols + ["logloss_mean", "brier_mean"]].copy()
    features["logloss_mean"] *= -1
    features["brier_mean"] *= -1
    coords, explained = pca_2d(features.to_numpy())
    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    size_map = {1: 12, 3: 22, 5: 34, 10: 48, 20: 66}
    ax.axhline(0, color=COLORS["grid"], linewidth=0.75, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.75, zorder=0)
    for label in METHOD_ORDER:
        block = sub[sub["label"].eq(label)]
        pts = block[["PC1", "PC2"]].to_numpy()
        covariance_ellipse(ax, pts, METHOD_COLORS[label])
        ax.scatter(block["PC1"], block["PC2"],
                   s=block["k"].map(size_map), color=METHOD_COLORS[label],
                   alpha=0.22, edgecolor="none", zorder=1)
        center = block[["PC1", "PC2"]].mean().to_numpy()
        ax.scatter(center[0], center[1], s=95, marker="D",
                   color=METHOD_COLORS[label], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(center[0], center[1] + 0.42, label.replace("Graph-MDWGAN", "Graph-MDWGAN"),
                ha="center", va="center", fontsize=6.8, fontweight="bold",
                color=METHOD_COLORS[label])
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7B8794",
               markeredgecolor="none", markersize=np.sqrt(size_map[k]),
               label=f"Top-{k}")
        for k in [1, 3, 5, 10, 20]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.53, 1.02), ncol=5, columnspacing=0.55,
              handletextpad=0.20)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    style_axes(ax)


def panel_d_processing_glyphs(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    clf_names = {"logreg": "Logistic", "svm": "SVM", "rf": "RF"}
    metrics = [("AUC", "auc_mean", "auc_sd"), ("AUPRC", "auprc_mean", "auprc_sd"),
               ("Bal. acc.", "balacc_mean", "balacc_sd"), ("MCC", "mcc_mean", "mcc_sd")]
    col_specs = [(clf, metric, mean_col, sd_col) for clf in clf_order for metric, mean_col, sd_col in metrics]
    y_order = METHOD_ORDER[::-1]
    for yi, label in enumerate(y_order):
        ax.text(-0.70, yi, label.replace("Graph-MDWGAN", "Graph"), ha="right", va="center",
                fontsize=7.0, fontweight="bold", color=METHOD_COLORS[label])
    for xi, (clf, metric, mean_col, sd_col) in enumerate(col_specs):
        if xi % 4 == 0:
            ax.axvline(xi - 0.55, color=COLORS["grid"], linewidth=0.9, zorder=0)
            ax.text(xi + 1.5, len(y_order) - 0.05, clf_names[clf],
                    ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
        vals = sub[sub["clf"].eq(clf)][mean_col]
        vmin, vmax = vals.min(), vals.max()
        for yi, label in enumerate(y_order):
            row = sub[(sub["label"].eq(label)) & (sub["clf"].eq(clf))].iloc[0]
            val = float(row[mean_col])
            sd = float(row[sd_col])
            norm = (val - 0.70) / 0.32 if metric != "MCC" else (val - 0.55) / 0.40
            alpha = float(np.clip(0.20 + 0.65 * norm, 0.18, 0.88))
            is_best = np.isclose(val, vmax)
            rect = FancyBboxPatch((xi - 0.40, yi - 0.32), 0.80, 0.64,
                                  boxstyle="round,pad=0.02,rounding_size=0.035",
                                  facecolor=METHOD_COLORS[label], edgecolor="white",
                                  linewidth=0.9, alpha=alpha)
            ax.add_patch(rect)
            ax.scatter(xi + 0.24, yi + 0.20, s=18 + 260 * sd,
                       color="white" if alpha > 0.55 else COLORS["ink"],
                       edgecolor="none", alpha=0.85)
            ax.text(xi, yi - 0.035, f"{val:.3f}", ha="center", va="center",
                    fontsize=5.6, fontweight="bold",
                    color="white" if alpha > 0.60 else COLORS["ink"])
            if is_best:
                ax.text(xi - 0.27, yi + 0.19, "*", ha="center", va="center",
                        fontsize=8.0, fontweight="bold",
                        color="white" if alpha > 0.60 else COLORS["ink"])
    ax.axvline(len(col_specs) - 0.45, color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.set_xlim(-0.95, len(col_specs) - 0.35)
    ax.set_ylim(-0.58, len(y_order) + 0.35)
    ax.set_xticks(range(len(col_specs)))
    ax.set_xticklabels([m for _, m, _, _ in col_specs], rotation=90)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=1)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(5.7)
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_e_duplicate_signature(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    ax.axvspan(-0.02, 0.08, color=COLORS["soft_green"], alpha=0.65, zorder=0)
    ax.axhspan(-0.02, 0.08, color=COLORS["soft_green"], alpha=0.65, zorder=0)
    for _, row in df.iterrows():
        label = str(row["label"])
        x = float(row["dup_to_real_mean"])
        y = float(row["dup_within_mean"])
        ax.scatter(x, y, s=150 if label == "ROS" else 90,
                   color=METHOD_COLORS[label], edgecolor="white",
                   linewidth=0.85, alpha=0.90, zorder=3)
        if label == "ROS":
            ax.text(0.93, 0.79, "ROS", ha="right", va="center",
                    fontsize=7.0, fontweight="bold", color=METHOD_COLORS[label])
    inset = ax.inset_axes([0.10, 0.56, 0.40, 0.34])
    offsets = {"Graph-MDWGAN": (0.018, 0.018), "RealOnly": (0.000, -0.012),
               "SMOTE": (-0.016, 0.006), "ROS": (0.0, 0.0)}
    for _, row in df[df["label"].ne("ROS")].iterrows():
        label = str(row["label"])
        x = float(row["dup_to_real_mean"]) + offsets[label][0]
        y = float(row["dup_within_mean"]) + offsets[label][1]
        inset.scatter(x, y, s=55, color=METHOD_COLORS[label],
                      edgecolor="white", linewidth=0.55, zorder=3)
        inset.text(x + 0.006, y + 0.002, label.replace("Graph-MDWGAN", "Graph"),
                   ha="left", va="center", fontsize=5.4, fontweight="bold",
                   color=METHOD_COLORS[label])
    inset.set_xlim(-0.035, 0.065)
    inset.set_ylim(-0.035, 0.065)
    inset.set_xticks([0])
    inset.set_yticks([0])
    inset.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    for tick in inset.get_xticklabels() + inset.get_yticklabels():
        tick.set_fontsize(5.6)
        tick.set_fontweight("bold")
    for spine in inset.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(COLORS["axis"])
    ax.plot([0.08, 0.18], [0.08, 0.56], color=COLORS["grid"], linewidth=0.7)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 0.96)
    ax.set_xlabel("Synthetic-real duplicate rate")
    ax.set_ylabel("Within-synthetic duplicate rate")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 0.85])
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    style_axes(ax)


def panel_f_lowdata_ridgelines(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["label"].eq("Graph-MDWGAN")) &
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg"))
    ].copy()
    piv = sub.pivot_table(index=["fold", "frac_real"], columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    fracs = sorted(piv["frac_real"].unique())
    x_grid = np.linspace(-0.28, 0.28, 260)
    for yi, frac in enumerate(fracs[::-1]):
        vals = piv[piv["frac_real"].eq(frac)]["delta"].to_numpy(dtype=float)
        dens = gaussian_kde(vals)(x_grid) if len(np.unique(np.round(vals, 5))) > 1 else np.zeros_like(x_grid)
        dens = dens / max(dens.max(), 1e-9) * 0.32
        base = yi
        ax.fill_between(x_grid, base, base + dens, color=COLORS["graph"], alpha=0.24, linewidth=0)
        ax.plot(x_grid, base + dens, color=COLORS["graph"], linewidth=1.1)
        ax.scatter(vals, np.full(len(vals), base - 0.035), s=9, color=COLORS["graph"],
                   alpha=0.35, edgecolor="none")
        mean, lo, hi = ci95(vals)
        ax.plot([lo, hi], [base + 0.41, base + 0.41], color=COLORS["ink"], linewidth=0.95)
        ax.scatter(mean, base + 0.41, marker="D", s=39, color=COLORS["graph"],
                   edgecolor="white", linewidth=0.65, zorder=3)
        ax.text(-0.300, base + 0.10, f"{int(frac * 100)}%", ha="right",
                va="center", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
        label_y = base + 0.56
        ax.plot([mean, mean + 0.018], [base + 0.41, label_y - 0.025],
                color=COLORS["graph"], linewidth=0.55, alpha=0.75, zorder=4)
        ax.text(mean + 0.022, label_y, f"{mean:+.3f}", ha="left", va="center",
                fontsize=5.9, fontweight="bold", color=COLORS["graph"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35})
    ax.axvline(0, color=COLORS["axis"], linewidth=0.8)
    ax.set_xlim(-0.31, 0.31)
    ax.set_ylim(-0.18, len(fracs) - 0.16)
    ax.set_xlabel("Augmented minus RealOnly balanced accuracy")
    ax.set_yticks([])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def panel_g_lowdata_paired_lanes(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg")) &
        (low_long["label"].isin(["Graph-MDWGAN", "ROS", "SMOTE"]))
    ].copy()
    piv = sub.pivot_table(index=["fold", "frac_real", "label"], columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    methods = ["Graph-MDWGAN", "ROS", "SMOTE"]
    fracs = sorted(piv["frac_real"].unique())
    for yi, label in enumerate(methods[::-1]):
        for xi, frac in enumerate(fracs):
            vals = piv[(piv["label"].eq(label)) & (piv["frac_real"].eq(frac))]["delta"].to_numpy(dtype=float)
            mean, lo, hi = ci95(vals)
            norm = np.clip((mean + 0.08) / 0.18, 0, 1)
            color = METHOD_COLORS[label]
            alpha = 0.18 + 0.65 * norm
            rect = FancyBboxPatch((xi - 0.39, yi - 0.33), 0.78, 0.66,
                                  boxstyle="round,pad=0.02,rounding_size=0.035",
                                  facecolor=color, edgecolor="white", linewidth=0.9,
                                  alpha=alpha)
            ax.add_patch(rect)
            ax.plot([xi - 0.22, xi + 0.22], [yi + 0.24, yi + 0.24],
                    color=COLORS["ink"], linewidth=0.80, alpha=0.80)
            ax.text(xi, yi - 0.04, f"{mean:+.3f}", ha="center", va="center",
                    fontsize=5.9, fontweight="bold",
                    color="white" if alpha > 0.52 else COLORS["ink"])
        ax.text(-0.58, yi, label.replace("Graph-MDWGAN", "Graph"),
                ha="right", va="center", fontsize=7.0, fontweight="bold",
                color=METHOD_COLORS[label])
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_yticks([])
    ax.set_xlabel("Fraction of real training samples")
    ax.set_xlim(-0.70, len(fracs) - 0.42)
    ax.set_ylim(-0.55, len(methods) - 0.45)
    ax.tick_params(length=0)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_h_dominance_orbits(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    stability = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "stability_mean_sd.csv")
    stability["label"] = stability["method"].map(METHOD_KEY).fillna(stability["method"])
    perf = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion"))
    ].copy()
    stab = stability[
        (stability["factor"].eq(8)) & (stability["fair"].eq("raw")) &
        (stability["scheme"].eq("fusion"))
    ][["factor", "method", "fair", "scheme", "k", "jaccard_mean"]]
    merged = perf.merge(stab, on=["factor", "method", "fair", "scheme", "k"], how="left")
    markers = {"logreg": "o", "svm": "D", "rf": "s"}
    for label in METHOD_ORDER:
        block = merged[merged["label"].eq(label)]
        for clf, marker in markers.items():
            ss = block[block["clf"].eq(clf)]
            ax.scatter(ss["jaccard_mean"], ss["auc_mean"], s=ss["k"] * 3.0 + 14,
                       marker=marker, color=METHOD_COLORS[label], edgecolor="white",
                       linewidth=0.55, alpha=0.62, zorder=2)
        target = block[(block["k"].eq(3)) & (block["clf"].eq("logreg"))].iloc[0]
        ax.text(target["jaccard_mean"] + 0.006, target["auc_mean"] + 0.006,
                label.replace("Graph-MDWGAN", "Graph"),
                ha="left", va="bottom", fontsize=6.0, fontweight="bold",
                color=METHOD_COLORS[label])
    ax.axhspan(0.95, 1.01, color=COLORS["soft_green"], alpha=0.38, zorder=0)
    ax.set_xlabel("Top-k ranking stability")
    ax.set_ylabel("Mean AUC")
    ax.set_xlim(0.18, 0.82)
    ax.set_ylim(0.84, 1.01)
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    handles = [Line2D([0], [0], marker=markers[c], color="none",
                      markerfacecolor="#7B8794", markeredgecolor="white",
                      markersize=5.2, label={"logreg": "Logistic", "svm": "SVM", "rf": "RF"}[c])
               for c in ["logreg", "svm", "rf"]]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.55, 1.02), ncol=3, handletextpad=0.25,
              columnspacing=0.8)
    style_axes(ax)


def panel_i_calibration_glyphs(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.23)
    sub = mean_df[
        (mean_df["factor"].eq(8)) & (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) & (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    clf_names = {"logreg": "Logistic", "svm": "Linear SVM", "rf": "Random forest"}
    for i, clf in enumerate(clf_order):
        subax = fig.add_subplot(subgs[0, i])
        block = sub[sub["clf"].eq(clf)].copy()
        for label in METHOD_ORDER:
            row = block[block["label"].eq(label)].iloc[0]
            subax.scatter(row["logloss_mean"], row["brier_mean"],
                          s=65 + 260 * (row["auc_mean"] - 0.88),
                          color=METHOD_COLORS[label], edgecolor="white",
                          linewidth=0.75, alpha=0.92, zorder=3)
            subax.text(row["logloss_mean"] + 0.010, row["brier_mean"] + 0.002,
                       label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real"),
                       ha="left", va="center", fontsize=5.5, fontweight="bold",
                       color=METHOD_COLORS[label])
        if i == 0:
            subax.annotate("lower is better", xy=(0.25, 0.080), xytext=(0.47, 0.128),
                           arrowprops={"arrowstyle": "->", "lw": 0.75, "color": COLORS["ink"]},
                           ha="left", va="center", fontsize=5.8, fontweight="bold",
                           color=COLORS["ink"])
            subax.set_ylabel("Mean Brier score")
        else:
            subax.set_yticklabels([])
        subax.text(0.03, 0.97, clf_names[clf], transform=subax.transAxes,
                   ha="left", va="top", fontsize=6.8, fontweight="bold",
                   color=COLORS["ink"])
        subax.set_xlabel("Mean log loss")
        subax.set_xlim(0.12, 1.70)
        subax.set_ylim(0.07, 0.15)
        subax.grid(color=COLORS["grid"], linewidth=0.48, alpha=0.70)
        style_axes(subax)


def panel_c_performance_embedding(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df[
        (mean_df["factor"].eq(8)) &
        (mean_df["scheme"].eq("fusion")) &
        (mean_df["k"].isin([1, 3, 5, 10]))
    ].copy()
    metric_cols = ["auc_mean", "auprc_mean", "balacc_mean", "mcc_mean", "logloss_mean", "brier_mean"]
    features = sub[metric_cols].copy()
    features[["logloss_mean", "brier_mean"]] *= -1
    for col in features.columns:
        lo, hi = np.nanpercentile(features[col], [2, 98])
        features[col] = features[col].clip(lo, hi)
    coords, explained = pca_2d(features.to_numpy())
    sub["PC1"] = coords[:, 0]
    sub["PC2"] = coords[:, 1]
    size_map = {1: 16, 3: 28, 5: 42, 10: 62}
    fair_markers = {"raw": "o", "dedup": "s", "fixunique": "^"}

    ax.axhline(0, color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.7, zorder=0)
    for label in METHOD_ORDER:
        block = sub[sub["label"].eq(label)]
        pts = block[["PC1", "PC2"]].to_numpy()
        covariance_ellipse(ax, pts, METHOD_COLORS[label])
        for fair, marker in fair_markers.items():
            ss = block[block["fair"].eq(fair)]
            ax.scatter(ss["PC1"], ss["PC2"],
                       s=ss["k"].map(size_map), marker=marker,
                       color=METHOD_COLORS[label], alpha=0.28,
                       edgecolor="white", linewidth=0.35, zorder=2)
        center = pts.mean(axis=0)
        ax.scatter(center[0], center[1], s=92, marker="D",
                   color=METHOD_COLORS[label], edgecolor="white", linewidth=0.8, zorder=4)

    centers = sub.groupby("label")[["PC1", "PC2"]].mean()
    label_offsets = {
        "Graph-MDWGAN": (0.18, 0.34, "left"),
        "RealOnly": (0.14, -0.36, "left"),
        "ROS": (0.16, 0.24, "left"),
        "SMOTE": (-0.18, -0.34, "right"),
    }
    for label, row in centers.iterrows():
        dx, dy, ha = label_offsets.get(label, (0.12, 0.26, "left"))
        ax.plot([row["PC1"], row["PC1"] + dx * 0.72],
                [row["PC2"], row["PC2"] + dy * 0.72],
                color=METHOD_COLORS[label], linewidth=0.55, alpha=0.65)
        ax.text(row["PC1"] + dx, row["PC2"] + dy,
                label.replace("Graph-MDWGAN", "Graph"),
                ha=ha, va="center", fontsize=6.5, fontweight="bold",
                color=METHOD_COLORS[label])

    shape_handles = [
        Line2D([0], [0], marker=fair_markers[fair], color="none",
               markerfacecolor="#7B8794", markeredgecolor="white",
               markersize=5.3, label=name)
        for fair, name in [("raw", "Raw"), ("dedup", "Dedup"), ("fixunique", "Fixunique")]
    ]
    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9AA5B1",
               markeredgecolor="white", markersize=np.sqrt(size_map[k]),
               label=f"Top-{k}")
        for k in [1, 3, 5, 10]
    ]
    ax.legend(handles=shape_handles + size_handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=7, columnspacing=0.48,
              handletextpad=0.18)
    qx = np.nanpercentile(np.abs(sub["PC1"]), 98) * 1.10
    qy = np.nanpercentile(np.abs(sub["PC2"]), 98) * 1.18
    ax.set_xlim(-qx, qx)
    ax.set_ylim(-qy, qy)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.58)
    style_axes(ax)


def panel_e_duplicate_signature(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.28, 1.0], wspace=0.12)
    left = fig.add_subplot(subgs[0, 0])
    right = fig.add_subplot(subgs[0, 1], sharey=left)
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    lanes = [("Synthetic-real", "dup_to_real_mean", 1.0), ("Within-synthetic", "dup_within_mean", 0.0)]
    offsets = {"Graph-MDWGAN": 0.12, "RealOnly": 0.04, "SMOTE": -0.04, "ROS": -0.12}

    for panel, xlim, ticks in [(left, (-0.006, 0.058), [0.00, 0.05]), (right, (0.82, 1.02), [0.85, 1.00])]:
        panel.axvspan(-0.006, 0.010, color=COLORS["soft_green"], alpha=0.66, zorder=0)
        for lane_label, col, base_y in lanes:
            panel.axhline(base_y, color=COLORS["grid"], linewidth=0.65, zorder=0)
            for _, row in df.iterrows():
                label = str(row["label"])
                value = float(row[col])
                if xlim[0] <= value <= xlim[1]:
                    y = base_y + offsets[label]
                    panel.plot([0 if xlim[0] < 0 else xlim[0], value], [y, y],
                               color=METHOD_COLORS[label], linewidth=1.15, alpha=0.70)
                    panel.scatter(value, y, s=58, color=METHOD_COLORS[label],
                                  edgecolor="white", linewidth=0.72, zorder=3)
                    text = label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real")
                    if panel is left:
                        panel.text(value + 0.0025, y, text, ha="left", va="center",
                                   fontsize=5.8, fontweight="bold", color=METHOD_COLORS[label])
                    else:
                        panel.text(value - 0.012, y, text if col == "dup_within_mean" or label == "ROS" else "",
                                   ha="right", va="center", fontsize=5.8, fontweight="bold",
                                   color=METHOD_COLORS[label])
        panel.set_xlim(*xlim)
        panel.set_ylim(-0.34, 1.34)
        panel.set_xticks(ticks)
        panel.grid(axis="x", color=COLORS["grid"], linewidth=0.48, alpha=0.72)
        style_axes(panel)
        panel.spines["top"].set_visible(False)

    left.set_yticks([1.0, 0.0])
    left.set_yticklabels([x[0] for x in lanes])
    for tick in left.get_yticklabels():
        tick.set_fontweight("bold")
    right.tick_params(labelleft=False, left=False)
    left.set_xlabel("Near-zero duplicate rate")
    right.set_xlabel("High duplicate rate")
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    kwargs = dict(marker=[(-1, -0.9), (1, 0.9)], markersize=6, linestyle="none",
                  color=COLORS["axis"], mec=COLORS["axis"], mew=0.8, clip_on=False)
    left.plot([1, 1], [0, 1], transform=left.transAxes, **kwargs)
    right.plot([0, 0], [0, 1], transform=right.transAxes, **kwargs)


def panel_g_lowdata_paired_lanes(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["k"].eq(10)) &
        (low_long["clf"].eq("logreg")) &
        (low_long["label"].isin(["Graph-MDWGAN", "ROS", "SMOTE"]))
    ].copy()
    piv = sub.pivot_table(index=["fold", "frac_real", "label"], columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    methods = ["Graph-MDWGAN", "ROS", "SMOTE"]
    fracs = sorted(piv["frac_real"].unique())
    x_vals = np.arange(len(fracs))
    scale = 2.25

    for lane_i, label in enumerate(methods[::-1]):
        base = lane_i
        color = METHOD_COLORS[label]
        block = piv[piv["label"].eq(label)]
        for fold, fold_block in block.groupby("fold"):
            fold_block = fold_block.sort_values("frac_real")
            y = base + fold_block["delta"].to_numpy(dtype=float) * scale
            ax.plot(x_vals, y, color=color, alpha=0.10, linewidth=0.65, zorder=1)
        means, lows, highs = [], [], []
        for frac in fracs:
            vals = block[block["frac_real"].eq(frac)]["delta"].to_numpy(dtype=float)
            mean, lo, hi = ci95(vals)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        ax.fill_between(x_vals, base + lows * scale, base + highs * scale,
                        color=color, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(x_vals, base + means * scale, color=color, linewidth=1.45, zorder=3)
        ax.scatter(x_vals, base + means * scale, s=32, color=color,
                   edgecolor="white", linewidth=0.65, zorder=4)
        ax.axhline(base, color=COLORS["axis"], linewidth=0.6, alpha=0.45, zorder=0)
        ax.text(-0.58, base + 0.01, label.replace("Graph-MDWGAN", "Graph"),
                ha="right", va="center", fontsize=6.8, fontweight="bold", color=color)
        for xi, mean in zip(x_vals, means):
            if abs(mean) >= 0.025 or (label == "Graph-MDWGAN" and xi == 0):
                ax.text(xi, base + mean * scale + (0.09 if mean >= 0 else -0.09),
                        f"{mean:+.3f}", ha="center",
                        va="bottom" if mean >= 0 else "top",
                        fontsize=5.4, fontweight="bold", color=color)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_yticks([])
    ax.set_xlim(-0.82, len(fracs) - 0.45)
    ax.set_ylim(-0.38, len(methods) - 0.58 + 0.72)
    ax.set_xlabel("Fraction of real training samples")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.48, alpha=0.70)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def panel_h_dominance_orbits(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    stability = pd.read_csv(CORRECTED_SECONDARY_RESULTS / "stability_mean_sd.csv")
    stability["label"] = stability["method"].map(METHOD_KEY).fillna(stability["method"])
    perf = mean_df[
        (mean_df["factor"].eq(8)) &
        (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion"))
    ].copy()
    stab = stability[
        (stability["factor"].eq(8)) &
        (stability["fair"].eq("raw")) &
        (stability["scheme"].eq("fusion"))
    ][["factor", "method", "fair", "scheme", "k", "jaccard_mean"]]
    merged = perf.merge(stab, on=["factor", "method", "fair", "scheme", "k"], how="left")
    markers = {"logreg": "o", "svm": "D", "rf": "s"}

    all_points = merged.dropna(subset=["jaccard_mean", "auc_mean"]).copy()
    frontier = []
    for idx, row in all_points.sort_values("jaccard_mean").iterrows():
        better = all_points[
            (all_points["jaccard_mean"].ge(row["jaccard_mean"])) &
            (all_points["auc_mean"].ge(row["auc_mean"])) &
            ((all_points["jaccard_mean"].gt(row["jaccard_mean"])) |
             (all_points["auc_mean"].gt(row["auc_mean"])))
        ]
        if better.empty:
            frontier.append(idx)
    front = all_points.loc[frontier].sort_values("jaccard_mean")

    for label in METHOD_ORDER:
        block = merged[merged["label"].eq(label)]
        for clf, marker in markers.items():
            ss = block[block["clf"].eq(clf)]
            ax.scatter(ss["jaccard_mean"], ss["auc_mean"], s=ss["k"] * 3.2 + 13,
                       marker=marker, color=METHOD_COLORS[label], edgecolor="white",
                       linewidth=0.55, alpha=0.60, zorder=2)
        target = block[(block["k"].eq(3)) & (block["clf"].eq("logreg"))].iloc[0]
        ax.scatter(target["jaccard_mean"], target["auc_mean"], s=88, marker="*",
                   color=METHOD_COLORS[label], edgecolor="white", linewidth=0.75, zorder=5)
        ax.text(target["jaccard_mean"] + 0.007, target["auc_mean"] + 0.005,
                label.replace("Graph-MDWGAN", "Graph"),
                ha="left", va="bottom", fontsize=5.8, fontweight="bold",
                color=METHOD_COLORS[label])
    if len(front) > 1:
        ax.plot(front["jaccard_mean"], front["auc_mean"], color=COLORS["ink"],
                linewidth=1.0, alpha=0.75, zorder=4)
        ax.text(front["jaccard_mean"].iloc[-1] - 0.005, front["auc_mean"].iloc[-1] - 0.012,
                "frontier", ha="right", va="top", fontsize=5.8,
                fontweight="bold", color=COLORS["ink"])
    ax.axhspan(0.95, 1.01, color=COLORS["soft_green"], alpha=0.34, zorder=0)
    ax.set_xlabel("Top-k ranking stability")
    ax.set_ylabel("Mean AUC")
    ax.set_xlim(0.18, 0.82)
    ax.set_ylim(0.84, 1.01)
    ax.grid(color=COLORS["grid"], linewidth=0.50, alpha=0.70)
    handles = [Line2D([0], [0], marker=markers[c], color="none",
                      markerfacecolor="#7B8794", markeredgecolor="white",
                      markersize=5.0, label={"logreg": "Logistic", "svm": "SVM", "rf": "RF"}[c])
               for c in ["logreg", "svm", "rf"]]
    handles.append(Line2D([0], [0], marker="*", color="none",
                          markerfacecolor="#7B8794", markeredgecolor="white",
                          markersize=7.0, label="Top-3 logistic"))
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=4, handletextpad=0.22,
              columnspacing=0.58)
    style_axes(ax)


def panel_i_calibration_glyphs(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.26)
    sub = mean_df[
        (mean_df["factor"].eq(8)) &
        (mean_df["fair"].eq("raw")) &
        (mean_df["scheme"].eq("fusion")) &
        (mean_df["k"].eq(3))
    ].copy()
    clf_order = ["logreg", "svm", "rf"]
    clf_names = {"logreg": "Logistic", "svm": "Linear SVM", "rf": "Random forest"}
    offsets = {
        "Graph-MDWGAN": (0.016, -0.004, "left"),
        "RealOnly": (0.014, 0.004, "left"),
        "ROS": (0.016, -0.004, "left"),
        "SMOTE": (0.014, 0.004, "left"),
    }
    clf_offsets = {
        ("svm", "SMOTE"): (0.18, -0.003, "left"),
        ("rf", "Graph-MDWGAN"): (0.030, -0.006, "left"),
        ("rf", "RealOnly"): (0.030, 0.007, "left"),
        ("rf", "ROS"): (0.020, -0.006, "left"),
    }
    for i, clf in enumerate(clf_order):
        subax = fig.add_subplot(subgs[0, i])
        block = sub[sub["clf"].eq(clf)].copy()
        xs = block["logloss_mean"].to_numpy()
        ys = block["brier_mean"].to_numpy()
        subax.fill_between([xs.min() - 0.05, np.median(xs)], ys.min() - 0.006, np.median(ys),
                           color=COLORS["soft_green"], alpha=0.34, linewidth=0, zorder=0)
        for label in METHOD_ORDER:
            row = block[block["label"].eq(label)].iloc[0]
            size = 56 + 360 * max(float(row["auc_mean"]) - 0.88, 0)
            subax.scatter(row["logloss_mean"], row["brier_mean"], s=size,
                          color=METHOD_COLORS[label], edgecolor="white",
                          linewidth=0.75, alpha=0.92, zorder=3)
            dx, dy, ha = clf_offsets.get((clf, label), offsets[label])
            subax.text(row["logloss_mean"] + dx, row["brier_mean"] + dy,
                       label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real"),
                       ha=ha, va="center", fontsize=5.4, fontweight="bold",
                       color=METHOD_COLORS[label])
        if i == 0:
            subax.annotate("lower is better", xy=(0.24, 0.079), xytext=(1.10, 0.083),
                           arrowprops={"arrowstyle": "->", "lw": 0.70, "color": COLORS["ink"]},
                           ha="left", va="center", fontsize=5.6, fontweight="bold",
                           color=COLORS["ink"])
            subax.set_ylabel("Mean Brier score")
        else:
            subax.set_yticklabels([])
        subax.text(0.03, 0.97, clf_names[clf], transform=subax.transAxes,
                   ha="left", va="top", fontsize=6.8, fontweight="bold",
                   color=COLORS["ink"])
        subax.set_xlabel("Mean log loss")
        subax.set_xlim(0.10, 1.72)
        subax.set_ylim(0.07, 0.15)
        subax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.70)
        style_axes(subax)


def panel_b_paired_auc_delta(ax: plt.Axes, long_df: pd.DataFrame) -> None:
    fig = ax.figure
    spec = ax.get_subplotspec()
    ax.remove()
    subgs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.18)
    sub = long_df[
        (long_df["factor"].eq(8)) &
        (long_df["fair"].eq("raw")) &
        (long_df["scheme"].eq("fusion")) &
        (long_df["k"].eq(3)) &
        (long_df["clf"].eq("logreg")) &
        (long_df["label"].isin(METHOD_ORDER))
    ].copy()
    piv = sub.pivot_table(index="fold", columns="label", values="auc")
    comps = ["RealOnly", "ROS", "SMOTE"]
    comp_names = {"RealOnly": "RealOnly", "ROS": "ROS", "SMOTE": "SMOTE"}
    for i, comp in enumerate(comps):
        subax = fig.add_subplot(subgs[0, i])
        graph = piv["Graph-MDWGAN"].to_numpy(dtype=float)
        other = piv[comp].to_numpy(dtype=float)
        delta = graph - other
        for g, o, d in zip(graph, other, delta):
            color = COLORS["graph"] if d >= 0 else METHOD_COLORS[comp]
            subax.plot([0, 1], [o, g], color=color, alpha=0.24,
                       linewidth=0.85, zorder=1)
            subax.scatter([0, 1], [o, g], s=13, color=color,
                          alpha=0.30, edgecolor="none", zorder=2)
        mean_o, lo_o, hi_o = ci95(other)
        mean_g, lo_g, hi_g = ci95(graph)
        subax.plot([-0.08, 0.08], [mean_o, mean_o], color=METHOD_COLORS[comp],
                   linewidth=1.55, zorder=4)
        subax.plot([0.92, 1.08], [mean_g, mean_g], color=COLORS["graph"],
                   linewidth=1.55, zorder=4)
        subax.plot([0, 0], [lo_o, hi_o], color=METHOD_COLORS[comp],
                   linewidth=1.0, zorder=3)
        subax.plot([1, 1], [lo_g, hi_g], color=COLORS["graph"],
                   linewidth=1.0, zorder=3)
        subax.scatter([0, 1], [mean_o, mean_g], marker="D", s=42,
                      color=[METHOD_COLORS[comp], COLORS["graph"]],
                      edgecolor="white", linewidth=0.7, zorder=5)
        mean_d, lo_d, hi_d = ci95(delta)
        subax.text(0.50, 1.012, f"{mean_d:+.3f}", ha="center", va="bottom",
                   fontsize=6.2, fontweight="bold", color=COLORS["ink"])
        subax.axhspan(0.95, 1.01, color=COLORS["soft_green"], alpha=0.36, zorder=0)
        subax.set_xlim(-0.28, 1.28)
        subax.set_ylim(0.64, 1.025)
        subax.set_xticks([0, 1])
        subax.set_xticklabels([comp_names[comp], "Graph"], rotation=0)
        for tick in subax.get_xticklabels():
            tick.set_fontsize(6.2)
            tick.set_fontweight("bold")
            tick.set_color(METHOD_COLORS[comp] if tick.get_text() != "Graph" else COLORS["graph"])
        if i == 0:
            subax.set_ylabel("Fold-wise AUC")
        else:
            subax.set_yticklabels([])
        subax.grid(axis="y", color=COLORS["grid"], linewidth=0.50, alpha=0.72)
        style_axes(subax)


def panel_c_performance_embedding(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df.copy()
    sub["protocol"] = (
        sub["factor"].astype(str) + "x|" + sub["fair"].astype(str) + "|" +
        sub["scheme"].astype(str) + "|k" + sub["k"].astype(str) + "|" +
        sub["clf"].astype(str)
    )
    piv = sub.pivot_table(index="protocol", columns="label", values="auc_mean")
    piv = piv.dropna(subset=["Graph-MDWGAN"]).sort_values("Graph-MDWGAN")
    x = np.arange(len(piv))
    window = 21
    ax.axhspan(0.95, 1.005, color=COLORS["soft_green"], alpha=0.36, zorder=0)
    ax.axhline(0.90, color=COLORS["grid"], linewidth=0.65, zorder=0)
    for label in METHOD_ORDER:
        y = piv[label].to_numpy(dtype=float)
        q = pd.Series(y).rolling(window, center=True, min_periods=5).mean().to_numpy()
        ax.scatter(x, y, s=8, color=METHOD_COLORS[label], alpha=0.08,
                   edgecolor="none", zorder=1)
        ax.plot(x, q, color=METHOD_COLORS[label], linewidth=1.55,
                alpha=0.95, zorder=3)
    label_y = {"Graph-MDWGAN": 1.002, "RealOnly": 0.994, "ROS": 0.986, "SMOTE": 0.978}
    for label in METHOD_ORDER:
        y = piv[label].to_numpy(dtype=float)
        q = pd.Series(y).rolling(window, center=True, min_periods=5).mean().to_numpy()
        y_end = q[-6] if np.isfinite(q[-6]) else y[-1]
        x0 = len(x) - 1
        x1 = len(x) + 8
        ax.plot([x0, x1 - 2], [y_end, label_y[label]], color=METHOD_COLORS[label],
                linewidth=0.70, alpha=0.78, clip_on=False)
        ax.text(x1, label_y[label],
                label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real"),
                ha="left", va="center", fontsize=6.4, fontweight="bold",
                color=METHOD_COLORS[label], clip_on=False)

    top_band = piv["Graph-MDWGAN"].ge(0.97)
    if top_band.any():
        start = int(np.argmax(top_band.to_numpy()))
        ax.axvspan(start, len(piv) - 1, color=COLORS["soft_blue"], alpha=0.20, zorder=0)
    ticks = np.linspace(0, len(piv) - 1, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t / (len(piv) - 1) * 100)}%" for t in ticks])
    ax.set_xlim(-3, len(piv) + 34)
    ax.set_ylim(0.68, 1.020)
    ax.set_xlabel("Sensitivity protocols ordered by Graph-MDWGAN AUC")
    ax.set_ylabel("Mean AUC")
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.62)
    style_axes(ax)


def panel_e_duplicate_signature(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    vertices = {
        "Synthetic-real uniqueness": np.array([0.08, 0.08]),
        "Within-synthetic uniqueness": np.array([0.92, 0.08]),
        "Performance": np.array([0.50, 0.82]),
    }

    def ternary_xy(perf: float, syn_unique: float, within_unique: float) -> np.ndarray:
        weights = np.array([syn_unique, within_unique, perf], dtype=float)
        weights = weights / weights.sum()
        return (
            weights[0] * vertices["Synthetic-real uniqueness"] +
            weights[1] * vertices["Within-synthetic uniqueness"] +
            weights[2] * vertices["Performance"]
        )

    tri = np.vstack([
        vertices["Synthetic-real uniqueness"],
        vertices["Within-synthetic uniqueness"],
        vertices["Performance"],
        vertices["Synthetic-real uniqueness"],
    ])
    ax.fill(tri[:, 0], tri[:, 1], color=COLORS["pale"], alpha=0.80, zorder=0)
    ax.plot(tri[:, 0], tri[:, 1], color=COLORS["axis"], linewidth=0.85, zorder=1)
    for frac in [0.25, 0.50, 0.75]:
        left = vertices["Synthetic-real uniqueness"] * (1 - frac) + vertices["Performance"] * frac
        right = vertices["Within-synthetic uniqueness"] * (1 - frac) + vertices["Performance"] * frac
        ax.plot([left[0], right[0]], [left[1], right[1]],
                color=COLORS["grid"], linewidth=0.55, alpha=0.70, zorder=0)
        bottom = vertices["Synthetic-real uniqueness"] * (1 - frac) + vertices["Within-synthetic uniqueness"] * frac
        right_edge = vertices["Performance"] * (1 - frac) + vertices["Within-synthetic uniqueness"] * frac
        ax.plot([bottom[0], right_edge[0]], [bottom[1], right_edge[1]],
                color=COLORS["grid"], linewidth=0.45, alpha=0.45, zorder=0)
        left_edge = vertices["Performance"] * (1 - frac) + vertices["Synthetic-real uniqueness"] * frac
        ax.plot([bottom[0], left_edge[0]], [bottom[1], left_edge[1]],
                color=COLORS["grid"], linewidth=0.45, alpha=0.45, zorder=0)

    label_offsets = {
        "Graph-MDWGAN": (0.00, 0.070, "center"),
        "RealOnly": (-0.110, -0.030, "right"),
        "SMOTE": (0.110, -0.030, "left"),
        "ROS": (0.000, -0.075, "center"),
    }
    for _, row in df.iterrows():
        label = str(row["label"])
        perf = (float(row["auc_mean"]) + float(row["auprc_mean"])) / 2
        syn_unique = 1 - float(row["dup_to_real_mean"])
        within_unique = 1 - float(row["dup_within_mean"])
        pt = ternary_xy(perf, syn_unique, within_unique)
        size = 95 + 180 * max(float(row["mcc_mean"]), 0)
        ax.scatter(pt[0], pt[1], s=size, color=METHOD_COLORS[label],
                   edgecolor="white", linewidth=0.85, alpha=0.92, zorder=3)
        ax.scatter(pt[0], pt[1], s=size * 1.85, color=METHOD_COLORS[label],
                   edgecolor="none", alpha=0.10, zorder=2)
        dx, dy, ha = label_offsets[label]
        ax.plot([pt[0], pt[0] + dx * 0.72], [pt[1], pt[1] + dy * 0.72],
                color=METHOD_COLORS[label], linewidth=0.55, alpha=0.65, zorder=2)
        ax.text(pt[0] + dx, pt[1] + dy,
                label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real"),
                ha=ha, va="center", fontsize=6.3, fontweight="bold",
                color=METHOD_COLORS[label], zorder=4)

    ax.text(vertices["Performance"][0], vertices["Performance"][1] + 0.065,
            "Performance", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=COLORS["ink"])
    ax.text(vertices["Synthetic-real uniqueness"][0] - 0.020,
            vertices["Synthetic-real uniqueness"][1] - 0.055,
            "Synthetic-real\nuniqueness", ha="right", va="top",
            fontsize=6.2, fontweight="bold", color=COLORS["ink"], linespacing=0.95)
    ax.text(vertices["Within-synthetic uniqueness"][0] + 0.020,
            vertices["Within-synthetic uniqueness"][1] - 0.055,
            "Within-synthetic\nuniqueness", ha="left", va="top",
            fontsize=6.2, fontweight="bold", color=COLORS["ink"], linespacing=0.95)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.04, 0.94)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_a_metric_forest(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    metric_colors = [COLORS["graph"], COLORS["sky"], COLORS["gold"], COLORS["rose"]]
    centers = np.linspace(0.15, 0.85, len(METHOD_ORDER))
    y0 = 0.55
    radii = [0.125, 0.098, 0.071, 0.044]
    for cx, (_, row) in zip(centers, df.iterrows()):
        label = str(row["label"])
        for radius, (metric, mean_col, _), color in zip(radii, METRICS, metric_colors):
            val = float(row[mean_col])
            ax.add_patch(Wedge((cx, y0), radius, 90, 450, width=0.014,
                               facecolor=COLORS["grid"], edgecolor="none", alpha=0.72))
            ax.add_patch(Wedge((cx, y0), radius, 90, 90 + 360 * val, width=0.014,
                               facecolor=color, edgecolor="none", alpha=0.96))
        ax.add_patch(Circle((cx, y0), 0.030, facecolor=METHOD_COLORS[label],
                            edgecolor="white", linewidth=0.7, alpha=0.95))
        ax.text(cx, 0.31, label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real"),
                ha="center", va="center", fontsize=7.1, fontweight="bold",
                color=METHOD_COLORS[label])
        ax.text(cx, 0.20, f"{row['auc_mean']:.3f}/{row['auprc_mean']:.3f}",
                ha="center", va="center", fontsize=5.9, fontweight="bold",
                color=COLORS["ink"])
        ax.text(cx, 0.13, f"{row['balacc_mean']:.3f}/{row['mcc_mean']:.3f}",
                ha="center", va="center", fontsize=5.9, fontweight="bold",
                color=COLORS["ink"])
    handles = [
        Line2D([0], [0], color=color, linewidth=2.5, label=metric)
        for (metric, _, _), color in zip(METRICS, metric_colors)
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 0.94), ncol=4, handlelength=1.0,
              columnspacing=0.72, handletextpad=0.28)
    ax.text(0.50, 0.06, "AUC/AUPRC and balanced accuracy/MCC shown below each glyph",
            ha="center", va="center", fontsize=5.8, fontweight="bold",
            color=COLORS["ink"])
    ax.set_xlim(0.03, 0.97)
    ax.set_ylim(0.02, 0.97)
    ax.axis("off")


def panel_c_performance_embedding(ax: plt.Axes, mean_df: pd.DataFrame) -> None:
    sub = mean_df.copy()
    sub["protocol"] = (
        sub["factor"].astype(str) + "x|" + sub["fair"].astype(str) + "|" +
        sub["scheme"].astype(str) + "|k" + sub["k"].astype(str) + "|" +
        sub["clf"].astype(str)
    )
    piv = sub.pivot_table(index="protocol", columns="label", values="auc_mean")
    piv = piv.dropna(subset=["Graph-MDWGAN"]).sort_values("Graph-MDWGAN")
    x = np.arange(len(piv))
    window = 21
    ax.axhspan(0.95, 1.005, color=COLORS["soft_green"], alpha=0.36, zorder=0)
    ax.axhline(0.90, color=COLORS["grid"], linewidth=0.65, zorder=0)
    handles = []
    for label in METHOD_ORDER:
        y = piv[label].to_numpy(dtype=float)
        q = pd.Series(y).rolling(window, center=True, min_periods=5).mean().to_numpy()
        ax.scatter(x, y, s=8, color=METHOD_COLORS[label], alpha=0.08,
                   edgecolor="none", zorder=1)
        ax.plot(x, q, color=METHOD_COLORS[label], linewidth=1.55,
                alpha=0.95, zorder=3)
        handles.append(Line2D([0], [0], color=METHOD_COLORS[label], linewidth=1.8,
                              label=label.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real")))
    top_band = piv["Graph-MDWGAN"].ge(0.97)
    if top_band.any():
        start = int(np.argmax(top_band.to_numpy()))
        ax.axvspan(start, len(piv) - 1, color=COLORS["soft_blue"], alpha=0.20, zorder=0)
    ticks = np.linspace(0, len(piv) - 1, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t / (len(piv) - 1) * 100)}%" for t in ticks])
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=4, handlelength=1.1,
              columnspacing=0.7, handletextpad=0.25)
    ax.set_xlim(-3, len(piv) + 3)
    ax.set_ylim(0.68, 1.020)
    ax.set_xlabel("Sensitivity protocols ordered by Graph-MDWGAN AUC")
    ax.set_ylabel("Mean AUC")
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.62)
    style_axes(ax)


def panel_e_duplicate_signature(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    x = np.arange(len(METHOD_ORDER))
    ax.axhline(0, color=COLORS["axis"], linewidth=0.8, zorder=0)
    ax.axhspan(0, 1.03, color=COLORS["soft_green"], alpha=0.22, zorder=0)
    ax.axhspan(-1.03, 0, color=COLORS["soft_orange"], alpha=0.18, zorder=0)
    for xi, (_, row) in zip(x, df.iterrows()):
        label = str(row["label"])
        perf = (float(row["auc_mean"]) + float(row["auprc_mean"])) / 2
        burden = (float(row["dup_to_real_mean"]) + float(row["dup_within_mean"])) / 2
        ax.plot([xi, xi], [0, perf], color=METHOD_COLORS[label],
                linewidth=2.2, solid_capstyle="round", alpha=0.88)
        ax.scatter(xi, perf, s=72, color=METHOD_COLORS[label],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([xi, xi], [0, -burden], color=METHOD_COLORS[label],
                linewidth=2.2, solid_capstyle="round", alpha=0.62)
        ax.scatter(xi, -burden, s=50, marker="s", color=METHOD_COLORS[label],
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(xi, perf + 0.055, f"{perf:.3f}", ha="center", va="bottom",
                fontsize=5.8, fontweight="bold", color=METHOD_COLORS[label])
        burden_txt = "0.000" if burden < 0.0005 else f"{burden:.3f}"
        ax.text(xi, -burden - 0.065 if burden > 0.03 else -0.085, burden_txt,
                ha="center", va="top", fontsize=5.8, fontweight="bold",
                color=METHOD_COLORS[label])
    ax.text(-0.56, 0.82, "Performance index", ha="left", va="center",
            fontsize=6.3, fontweight="bold", color=COLORS["ink"])
    ax.text(-0.56, -0.82, "Duplicate burden", ha="left", va="center",
            fontsize=6.3, fontweight="bold", color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real") for m in METHOD_ORDER])
    for tick, method in zip(ax.get_xticklabels(), METHOD_ORDER):
        tick.set_fontweight("bold")
        tick.set_color(METHOD_COLORS[method])
    ax.set_xlim(-0.62, len(METHOD_ORDER) - 0.38)
    ax.set_ylim(-1.04, 1.08)
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.48, alpha=0.70)
    style_axes(ax)


def panel_f_lowdata_ridgelines(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    sub = low_long[
        (low_long["label"].eq("Graph-MDWGAN")) &
        (low_long["k"].eq(10)) & (low_long["clf"].eq("logreg"))
    ].copy()
    piv = sub.pivot_table(index=["fold", "frac_real"], columns="train_mode", values="bacc").reset_index()
    piv["delta"] = piv["aug"] - piv["real"]
    fracs = sorted(piv["frac_real"].unique())
    rng = np.random.default_rng(3606)
    for xi, frac in enumerate(fracs):
        vals = piv[piv["frac_real"].eq(frac)]["delta"].to_numpy(dtype=float)
        order = np.argsort(vals)
        vals = vals[order]
        y = np.linspace(-0.28, 0.28, len(vals)) + rng.normal(0, 0.004, len(vals))
        for yy, val in zip(y, vals):
            color = COLORS["graph"] if val >= 0 else COLORS["rose"]
            ax.plot([xi - 0.30, xi + 0.30], [yy, yy], color=COLORS["grid"],
                    linewidth=0.45, alpha=0.55, zorder=0)
            ax.scatter(xi + val * 1.55, yy, s=15, color=color,
                       alpha=0.70, edgecolor="white", linewidth=0.35, zorder=2)
        mean, lo, hi = ci95(vals)
        ax.plot([xi + lo * 1.55, xi + hi * 1.55], [0.40, 0.40],
                color=COLORS["ink"], linewidth=0.95, zorder=3)
        ax.scatter(xi + mean * 1.55, 0.40, marker="D", s=39,
                   color=COLORS["graph"], edgecolor="white", linewidth=0.65, zorder=4)
        ax.text(xi + mean * 1.55 + 0.055, 0.48, f"{mean:+.3f}",
                ha="left", va="center", fontsize=5.8, fontweight="bold",
                color=COLORS["graph"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35})
        ax.axvline(xi, color=COLORS["axis"], linewidth=0.55, alpha=0.50, zorder=0)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_xlabel("Fraction of real training samples")
    ax.set_ylabel("Fold-level delta barcode")
    ax.set_xlim(-0.55, len(fracs) - 0.45)
    ax.set_ylim(-0.39, 0.57)
    ax.set_yticks([])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.48, alpha=0.70)
    style_axes(ax)
    ax.spines["left"].set_visible(False)


def panel_a_metric_forest(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline)
    y_base = np.arange(len(METHOD_ORDER))[::-1]
    offsets = np.array([0.30, 0.10, -0.10, -0.30])
    metric_colors = [COLORS["graph"], COLORS["sky"], COLORS["gold"], COLORS["rose"]]
    for y, label in zip(y_base, METHOD_ORDER):
        ax.hlines(y, 0.635, 1.055, color=COLORS["grid"], linewidth=0.6, zorder=0)
        row = df[df["label"].eq(label)].iloc[0]
        for j, (metric, mean_col, sd_col) in enumerate(METRICS):
            yj = y + offsets[j]
            mean = float(row[mean_col])
            sd = float(row[sd_col])
            lo = max(0, mean - sd)
            hi = min(1.02, mean + sd)
            ax.plot([lo, hi], [yj, yj], color=metric_colors[j], linewidth=1.35,
                    solid_capstyle="round", alpha=0.90)
            ax.scatter(mean, yj, s=38, color=metric_colors[j], edgecolor="white",
                       linewidth=0.65, zorder=3)
            ax.text(1.028, yj, f"{mean:.3f}", ha="left", va="center",
                    fontsize=6.3, fontweight="bold", color=metric_colors[j])
    ax.set_xlim(0.58, 1.085)
    ax.set_ylim(-0.75, len(METHOD_ORDER) - 0.25)
    ax.set_yticks(y_base)
    ax.set_yticklabels(METHOD_ORDER)
    ax.tick_params(axis="y", length=0, pad=10)
    for tick, label in zip(ax.get_yticklabels(), METHOD_ORDER):
        tick.set_fontweight("bold")
        tick.set_color(METHOD_COLORS[label])
    ax.set_xlabel("Cross-validated metric value")
    ax.set_xticks([0.70, 0.80, 0.90, 1.00])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    handles = [
        Line2D([0], [0], marker="o", color=color, label=metric, markersize=5.2,
               markerfacecolor=color, markeredgecolor="white", linewidth=1.2)
        for (metric, _, _), color in zip(METRICS, metric_colors)
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.50, 1.02), ncol=4, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.28)
    style_axes(ax)


def panel_e_duplicate_signature(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline_ordered(baseline).set_index("label").loc[METHOD_ORDER].reset_index()
    x = np.arange(len(METHOD_ORDER))
    ax.axhline(0, color=COLORS["axis"], linewidth=0.8, zorder=0)
    ax.axhspan(0, 1.03, color=COLORS["soft_green"], alpha=0.22, zorder=0)
    ax.axhspan(-1.03, 0, color=COLORS["soft_orange"], alpha=0.18, zorder=0)
    for xi, (_, row) in zip(x, df.iterrows()):
        label = str(row["label"])
        perf = (float(row["auc_mean"]) + float(row["auprc_mean"])) / 2
        burden = (float(row["dup_to_real_mean"]) + float(row["dup_within_mean"])) / 2
        ax.plot([xi, xi], [0, perf], color=METHOD_COLORS[label],
                linewidth=2.2, solid_capstyle="round", alpha=0.88)
        ax.scatter(xi, perf, s=72, color=METHOD_COLORS[label],
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([xi, xi], [0, -burden], color=METHOD_COLORS[label],
                linewidth=2.2, solid_capstyle="round", alpha=0.62)
        ax.scatter(xi, -burden, s=50, marker="s", color=METHOD_COLORS[label],
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.text(xi, perf + 0.055, f"{perf:.3f}", ha="center", va="bottom",
                fontsize=5.8, fontweight="bold", color=METHOD_COLORS[label])
        burden_txt = "0.000" if burden < 0.0005 else f"{burden:.3f}"
        if burden > 0.03:
            ax.plot([xi, xi + 0.20], [-burden, -burden + 0.10],
                    color=METHOD_COLORS[label], linewidth=0.55, alpha=0.75, zorder=4)
            ax.text(xi + 0.23, -burden + 0.12, burden_txt,
                    ha="left", va="center", fontsize=5.8, fontweight="bold",
                    color=METHOD_COLORS[label],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.35})
        else:
            ax.text(xi, -0.085, burden_txt, ha="center", va="top",
                    fontsize=5.8, fontweight="bold", color=METHOD_COLORS[label],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.25})
    ax.text(-0.56, 0.82, "Performance index", ha="left", va="center",
            fontsize=6.3, fontweight="bold", color=COLORS["ink"])
    ax.text(-0.56, -0.82, "Duplicate burden", ha="left", va="center",
            fontsize=6.3, fontweight="bold", color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Graph-MDWGAN", "Graph").replace("RealOnly", "Real") for m in METHOD_ORDER])
    for tick, method in zip(ax.get_xticklabels(), METHOD_ORDER):
        tick.set_fontweight("bold")
        tick.set_color(METHOD_COLORS[method])
    ax.set_xlim(-0.62, len(METHOD_ORDER) - 0.18)
    ax.set_ylim(-1.04, 1.08)
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.48, alpha=0.70)
    style_axes(ax)


def make_figures() -> None:
    baseline, final_mean, final_long, low_mean, low_long = load_data()
    baseline.to_csv(OUT / "figure_3_6_baseline_source.csv", index=False)
    final_mean.to_csv(OUT / "figure_3_6_final_mean_source.csv", index=False)
    final_long.to_csv(OUT / "figure_3_6_final_long_source.csv", index=False)
    low_mean.to_csv(OUT / "figure_3_6_lowdata_mean_source.csv", index=False)
    low_long.to_csv(OUT / "figure_3_6_lowdata_long_source.csv", index=False)

    panels = [
        ("premium_panel_A_metric_forest", lambda ax: panel_a_metric_forest(ax, baseline), (5.8, 3.4)),
        ("premium_panel_B_paired_auc_delta", lambda ax: panel_b_paired_auc_delta(ax, final_long), (4.9, 3.2)),
        ("premium_panel_C_performance_embedding", lambda ax: panel_c_performance_embedding(ax, final_mean), (5.5, 3.55)),
        ("premium_panel_D_processing_glyphs", lambda ax: panel_d_processing_glyphs(ax, final_mean), (6.8, 3.25)),
        ("premium_panel_E_duplicate_signature", lambda ax: panel_e_duplicate_signature(ax, baseline), (5.2, 3.05)),
        ("premium_panel_F_lowdata_ridgelines", lambda ax: panel_f_lowdata_ridgelines(ax, low_long), (5.5, 3.35)),
        ("premium_panel_G_lowdata_paired_lanes", lambda ax: panel_g_lowdata_paired_lanes(ax, low_long), (5.4, 3.20)),
        ("premium_panel_H_dominance_orbits", lambda ax: panel_h_dominance_orbits(ax, final_mean), (5.2, 3.25)),
        ("premium_panel_I_calibration_glyphs", lambda ax: panel_i_calibration_glyphs(ax, final_mean), (6.6, 3.35)),
    ]

    for stem, drawer, size in panels:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax)
        save_all(fig, stem)

    fig = plt.figure(figsize=(14.2, 15.4))
    gs = fig.add_gridspec(
        4, 6,
        height_ratios=[1.06, 1.00, 1.02, 1.03],
        width_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.64,
        hspace=0.58,
    )
    axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
        fig.add_subplot(gs[3, 0:3]),
        fig.add_subplot(gs[3, 3:6]),
    ]
    drawers = [
        lambda ax: panel_a_metric_forest(ax, baseline),
        lambda ax: panel_b_paired_auc_delta(ax, final_long),
        lambda ax: panel_c_performance_embedding(ax, final_mean),
        lambda ax: panel_d_processing_glyphs(ax, final_mean),
        lambda ax: panel_e_duplicate_signature(ax, baseline),
        lambda ax: panel_f_lowdata_ridgelines(ax, low_long),
        lambda ax: panel_g_lowdata_paired_lanes(ax, low_long),
        lambda ax: panel_h_dominance_orbits(ax, final_mean),
        lambda ax: panel_i_calibration_glyphs(ax, final_mean),
    ]
    for ax, drawer in zip(axes, drawers):
        drawer(ax)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_3_6_robustness_combined.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    make_figures()
    print(f"Saved Figure 3.6 robustness panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
