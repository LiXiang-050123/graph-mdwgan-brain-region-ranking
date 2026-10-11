from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, sem, t

from analysis_paths import (
    CODE_ROOT,
    CORRECTED_SECONDARY_RESULTS,
    ROBUSTNESS_SOURCES,
    ROS_FIXUNIQUE_RESULTS,
)


ROOT = CODE_ROOT
SOURCE = ROBUSTNESS_SOURCES
OUT = ROOT / "figure_3_6_supplement_4panel_premium"
OUT.mkdir(exist_ok=True)
CORRECTED_RESULTS = CORRECTED_SECONDARY_RESULTS
CORRECTED_ROS_SUPPLEMENT = ROS_FIXUNIQUE_RESULTS

COLORS = {
    "Graph-MDWGAN": "#2B9A92",
    "RealOnly": "#21364D",
    "ROS": "#C97943",
    "SMOTE": "#7B68B3",
    "aug": "#2B9A92",
    "real": "#21364D",
    "axis": "#2E3338",
    "line": "#DDE5EA",
    "pale": "#F4F8F8",
    "text": "#17304A",
    "gray": "#9BA5AF",
}
METHOD_ORDER = ["Graph-MDWGAN", "RealOnly", "ROS", "SMOTE"]
FAIR_ORDER = ["raw", "dedup", "fixunique"]
CLF_LABELS = {"logreg": "Logistic", "svm": "Linear SVM", "rf": "Random forest"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Arial")


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT / f"{stem}_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    if len(values) < 2:
        return mean, mean, mean
    hw = float(t.ppf(0.975, len(values) - 1) * sem(values, nan_policy="omit"))
    return mean, mean - hw, mean + hw


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(CORRECTED_RESULTS / "perf_long.csv")
    raw["label"] = raw["method"].replace({"Ours": "Graph-MDWGAN"})

    # Graph-MDWGAN, RealOnly, and SMOTE have no exact duplicate rows, so
    # deduplication and uniform inverse-multiplicity weighting leave their
    # fitted data unchanged. ROS deduplication reduces to RealOnly training.
    unchanged = raw[raw["method"].isin(["Ours", "RealOnly", "SMOTE"])].copy()
    derived = []
    for fair in ("dedup", "fixunique"):
        copy = unchanged.copy()
        copy["fair"] = fair
        derived.append(copy)

    ros_dedup = raw[raw["method"].eq("RealOnly")].copy()
    ros_dedup["method"] = "ROS"
    ros_dedup["label"] = "ROS"
    ros_dedup["fair"] = "dedup"

    ros_fixunique = pd.read_csv(CORRECTED_ROS_SUPPLEMENT / "perf_long.csv")
    ros_fixunique = ros_fixunique[ros_fixunique["fair"].eq("fixunique")].copy()
    ros_fixunique["label"] = "ROS"

    final_long = pd.concat(
        [raw, *derived, ros_dedup, ros_fixunique], ignore_index=True
    )
    baseline = pd.read_csv(SOURCE / "figure_3_6_baseline_source.csv")
    low_long = pd.read_csv(SOURCE / "figure_3_6_lowdata_long_source.csv")
    return final_long, baseline, low_long


def draw_box_points(ax: plt.Axes, x: float, vals: np.ndarray, color: str, width: float = 0.18) -> None:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return
    if len(vals) >= 4 and np.nanstd(vals) > 1e-8:
        try:
            pad = max(0.015, 0.06 * (float(np.nanmax(vals)) - float(np.nanmin(vals))))
            y_grid = np.linspace(float(np.nanmin(vals)) - pad, float(np.nanmax(vals)) + pad, 120)
            dens = gaussian_kde(vals)(y_grid)
            dens = dens / dens.max() * width * 0.70
            ax.fill_betweenx(y_grid, x + width * 0.58, x + width * 0.58 + dens,
                             color=color, alpha=0.115, linewidth=0, zorder=1)
            ax.plot(x + width * 0.58 + dens, y_grid, color=color, alpha=0.55, lw=0.75, zorder=2)
        except Exception:
            pass
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi = np.min(vals), np.max(vals)
    ax.plot([x, x], [q1, q3], color=color, lw=4.2, alpha=0.25,
            solid_capstyle="round", zorder=3)
    ax.plot([x, x], [lo, hi], color=color, lw=0.85, alpha=0.72, zorder=3)
    ax.plot([x - width * 0.34, x + width * 0.34], [med, med],
            color=color, lw=1.45, zorder=5)
    ax.plot([x - width * 0.22, x + width * 0.22], [lo, lo],
            color=color, lw=0.75, alpha=0.70, zorder=4)
    ax.plot([x - width * 0.22, x + width * 0.22], [hi, hi],
            color=color, lw=0.75, alpha=0.70, zorder=4)
    rng = np.random.default_rng(int((x + 10) * 1000))
    ax.scatter(rng.normal(x - width * 0.30, width / 8, len(vals)), vals, s=9, color=color, alpha=0.30,
               edgecolor="none", zorder=5)
    ax.scatter([x], [np.nanmean(vals)], marker="D", s=23, color=color, edgecolor="white",
               linewidth=0.45, zorder=6)


def panel_a_preprocessing_box(ax: plt.Axes, final_long: pd.DataFrame) -> None:
    df = final_long[
        (final_long["factor"].eq(8))
        & (final_long["scheme"].eq("fusion"))
        & (final_long["k"].eq(3))
        & (final_long["clf"].eq("logreg"))
        & (final_long["label"].isin(METHOD_ORDER))
    ].copy()
    offsets = np.linspace(-0.27, 0.27, len(METHOD_ORDER))
    for i, fair in enumerate(FAIR_ORDER):
        for off, method in zip(offsets, METHOD_ORDER):
            vals = df[(df["fair"].eq(fair)) & (df["label"].eq(method))]["balacc"].to_numpy()
            draw_box_points(ax, i + off, vals, COLORS[method], width=0.12)
    ax.set_xticks(range(len(FAIR_ORDER)))
    ax.set_xticklabels(["Raw", "Dedup", "Fixunique"])
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Data-processing condition")
    ax.set_ylim(0.50, 1.035)
    handles = [Line2D([0], [0], marker="s", color=COLORS[m], markerfacecolor=COLORS[m],
                      linestyle="none", markersize=5, label=m) for m in METHOD_ORDER]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=0.75)
    ax.grid(axis="y", color=COLORS["line"], lw=0.55, alpha=0.70)
    style_axes(ax)


def panel_b_duplicate_audit(ax: plt.Axes, baseline: pd.DataFrame) -> None:
    df = baseline.copy()
    metrics = [("dup_to_real_mean", "Synthetic vs\nreal training"), ("dup_within_mean", "Within synthetic\nsamples")]
    x = np.arange(len(metrics))
    offsets = np.linspace(-0.24, 0.24, 3)
    methods = ["Graph-MDWGAN", "SMOTE", "ROS"]
    for off, method in zip(offsets, methods):
        sub = df[df["label"].eq(method)]
        vals = [float(sub[col].iloc[0]) * 100 for col, _ in metrics]
        ax.plot(x + off, vals, color=COLORS[method], lw=1.0, alpha=0.75)
        ax.scatter(x + off, vals, s=42, color=COLORS[method], edgecolor="white", lw=0.6, zorder=3)
        for xx, yy in zip(x + off, vals):
            label = "0%" if yy < 0.01 else f"{yy:.1f}%"
            ax.text(xx, yy + 3.0, label, ha="center", va="bottom", fontsize=7.2,
                    fontweight="bold", color=COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_ylabel("Duplicate rate")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    handles = [Line2D([0], [0], marker="o", color=COLORS[m], label=m,
                      markerfacecolor=COLORS[m], markersize=4.8) for m in methods]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=0.85)
    ax.grid(axis="y", color=COLORS["line"], lw=0.55, alpha=0.70)
    style_axes(ax)


def panel_c_lowdata_mcc_boxes(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    df = low_long[(low_long["method"].eq("Ours")) & (low_long["clf"].eq("logreg"))].copy()
    fractions = sorted(df["frac_real"].unique())
    colors = {"real": COLORS["RealOnly"], "aug": COLORS["Graph-MDWGAN"]}
    offsets = {"real": -0.11, "aug": 0.11}
    for i, frac in enumerate(fractions):
        for mode in ["real", "aug"]:
            vals = df[(df["frac_real"].eq(frac)) & (df["train_mode"].eq(mode))]["mcc"].to_numpy()
            draw_box_points(ax, i + offsets[mode], vals, colors[mode], width=0.15)
    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{int(x * 100)}%" for x in fractions])
    ax.set_xlabel("Fraction of real training samples")
    ax.set_ylabel("MCC")
    ax.set_ylim(0.10, 1.04)
    handles = [
        Line2D([0], [0], marker="s", color=COLORS["RealOnly"], label="Real only",
               linestyle="none", markerfacecolor=COLORS["RealOnly"], markersize=5),
        Line2D([0], [0], marker="s", color=COLORS["Graph-MDWGAN"], label="Graph-augmented",
               linestyle="none", markerfacecolor=COLORS["Graph-MDWGAN"], markersize=5),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", color=COLORS["line"], lw=0.55, alpha=0.70)
    style_axes(ax)


def panel_d_lowdata_logloss_line(ax: plt.Axes, low_long: pd.DataFrame) -> None:
    df = low_long[(low_long["method"].eq("Ours")) & (low_long["clf"].eq("logreg"))].copy()
    fractions = sorted(df["frac_real"].unique())
    for mode, color, label in [("real", COLORS["RealOnly"], "Real only"), ("aug", COLORS["Graph-MDWGAN"], "Graph-augmented")]:
        for _, sub in df[df["train_mode"].eq(mode)].groupby("fold"):
            sub = sub.sort_values("frac_real")
            ax.plot(sub["frac_real"], sub["logloss"], color=color, lw=0.45, alpha=0.13, zorder=1)
        means, lows, highs = [], [], []
        for frac in fractions:
            vals = df[(df["train_mode"].eq(mode)) & (df["frac_real"].eq(frac))]["logloss"].to_numpy()
            mean, lo, hi = mean_ci(vals)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        ax.plot(fractions, means, color=color, lw=1.9, label=label)
        ax.errorbar(fractions, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
                    fmt="o", ms=4.3, color=color, mec="white", mew=0.5, capsize=2.0, lw=1.0)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{int(x * 100)}%" for x in fractions])
    ax.set_xlabel("Fraction of real training samples")
    ax.set_ylabel("Log loss")
    ax.set_ylim(0.0, 0.82)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(color=COLORS["line"], lw=0.55, alpha=0.70)
    style_axes(ax)


def panel_e_loss_distributions(ax: plt.Axes, final_long: pd.DataFrame) -> None:
    df = final_long[
        (final_long["factor"].eq(8))
        & (final_long["fair"].eq("raw"))
        & (final_long["scheme"].eq("fusion"))
        & (final_long["k"].eq(3))
        & (final_long["clf"].eq("logreg"))
        & (final_long["label"].isin(METHOD_ORDER))
    ].copy()
    positions = []
    labels = []
    for i, method in enumerate(METHOD_ORDER):
        base = i * 0.9
        positions.append(base - 0.13)
        positions.append(base + 0.13)
        labels.append(method)
        for pos, metric, alpha in [(base - 0.13, "logloss", 0.22), (base + 0.13, "brier", 0.38)]:
            vals = df[df["label"].eq(method)][metric].to_numpy()
            draw_box_points(ax, pos, vals, COLORS[method], width=0.16)
    ax.set_xticks([i * 0.9 for i in range(len(METHOD_ORDER))])
    ax.set_xticklabels(METHOD_ORDER, rotation=16, ha="right")
    ax.set_ylabel("Loss value")
    ax.set_ylim(0, 1.05)
    handles = [
        Line2D([0], [0], marker="s", color=COLORS["axis"], markerfacecolor="white",
               linestyle="none", markersize=5, label="Log loss"),
        Line2D([0], [0], marker="s", color=COLORS["gray"], markerfacecolor=COLORS["gray"],
               linestyle="none", markersize=5, label="Brier score"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.grid(axis="y", color=COLORS["line"], lw=0.55, alpha=0.70)
    style_axes(ax)


def add_panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.12, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=13.5, fontweight="bold", color="black")


def make_figures() -> None:
    final_long, baseline, low_long = load_data()
    final_long.to_csv(OUT / "supplement_final_long_source.csv", index=False)
    baseline.to_csv(OUT / "supplement_baseline_source.csv", index=False)
    low_long.to_csv(OUT / "supplement_lowdata_long_source.csv", index=False)

    panels = [
        ("supp_panel_A_preprocessing_balanced_accuracy", lambda ax: panel_a_preprocessing_box(ax, final_long), (5.6, 3.55)),
        ("supp_panel_B_lowdata_mcc_distribution", lambda ax: panel_c_lowdata_mcc_boxes(ax, low_long), (5.1, 3.55)),
        ("supp_panel_C_lowdata_logloss", lambda ax: panel_d_lowdata_logloss_line(ax, low_long), (5.1, 3.55)),
        ("supp_panel_D_loss_distribution", lambda ax: panel_e_loss_distributions(ax, final_long), (5.45, 3.55)),
    ]
    for stem, func, size in panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax)
        save_all(fig, stem)

    fig = plt.figure(figsize=(11.6, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.70, wspace=0.48)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    funcs = [
        lambda ax: panel_a_preprocessing_box(ax, final_long),
        lambda ax: panel_c_lowdata_mcc_boxes(ax, low_long),
        lambda ax: panel_d_lowdata_logloss_line(ax, low_long),
        lambda ax: panel_e_loss_distributions(ax, final_long),
    ]
    for ax, func, letter in zip(axes, funcs, "ABCD"):
        func(ax)
        add_panel_letter(ax, letter)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_3_6_supplement_4panel_premium.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        OUT / "figure_3_6_supplement_4panel_premium_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    zip_path = ROOT / "figure_3_6_supplement_4panel_premium_300dpi_tiff.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for path in sorted(OUT.glob("*")):
            if path.suffix.lower() in {".tif", ".tiff", ".png", ".pdf", ".svg", ".csv"}:
                zf.write(path, arcname=f"{OUT.name}/{path.name}")


def main() -> None:
    set_style()
    make_figures()
    print(f"Saved Figure 3.6 supplement panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
