from __future__ import annotations

import gc
import shutil
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import plot_figure_3_7_literature_concordance as base
from analysis_paths import CODE_ROOT


ROOT = CODE_ROOT
OUT = ROOT / "figure_3_7_ABCFG_clear_300dpi_tiff"
OUT.mkdir(exist_ok=True)


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    return base.load_data()


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


def panel_b_rank_braid_clear(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s.copy()
    x_ml, x_lit = 0.0, 1.0
    for _, row in df.iterrows():
        region = str(row["brain_region"])
        is_candidate = region in base.CANDIDATES
        color = base.REGION_COLORS.get(region, base.COLORS["neutral"]) if is_candidate else "#C6D1D8"
        ax.plot(
            [x_ml, x_lit],
            [row["ml_rank"], row["literature_composite_rank"]],
            color=color,
            linewidth=2.2 if is_candidate else 0.9,
            alpha=0.96 if is_candidate else 0.34,
            zorder=3 if is_candidate else 1,
        )
        ax.scatter(
            [x_ml, x_lit],
            [row["ml_rank"], row["literature_composite_rank"]],
            s=48 if is_candidate else 18,
            color=color,
            edgecolor="white",
            linewidth=0.75 if is_candidate else 0.35,
            alpha=0.96 if is_candidate else 0.42,
            zorder=4 if is_candidate else 2,
        )
        if is_candidate:
            ax.text(
                x_ml - 0.055,
                row["ml_rank"],
                region,
                ha="right",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                color=color,
            )
            ax.text(
                x_lit + 0.055,
                row["literature_composite_rank"],
                region,
                ha="left",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                color=color,
            )
    for x, lab in [(x_ml, "ML rank"), (x_lit, "Literature rank")]:
        ax.axvline(x, color=base.COLORS["axis"], linewidth=1.05)
        ax.text(
            x,
            0.05,
            lab,
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
            color=base.COLORS["ink"],
        )
    ax.axhspan(0.5, 3.5, color=base.COLORS["band_green"], alpha=0.70, zorder=0)
    ax.axhspan(3.5, 7.5, color=base.COLORS["band_orange"], alpha=0.38, zorder=0)
    ax.set_xlim(-0.36, 1.36)
    ax.set_ylim(18.7, 0.35)
    ax.set_yticks([1, 3, 5, 10, 15, 18])
    ax.set_ylabel("Rank, lower is higher priority")
    ax.set_xticks([])
    ax.grid(axis="y", color=base.COLORS["grid"], linewidth=0.45, alpha=0.55)
    handles = [
        Line2D([0], [0], color=base.COLORS["ink"], lw=0, marker="o",
               markerfacecolor=base.COLORS["convergent"], markeredgecolor="white",
               label="Prioritized regions"),
        Line2D([0], [0], color="#C6D1D8", lw=1.0, label="Other regions"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        ncol=2,
        handlelength=1.4,
        columnspacing=1.2,
    )
    base.style_axes(ax, bottom=False)


def save_panel(label: str, stem: str, drawer, size: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=size)
    drawer(ax)
    fig.savefig(
        OUT / f"Fig3_7_panel_{label}_{stem}_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        OUT / f"Fig3_7_panel_{label}_{stem}_preview.png",
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
        OUT / f"Fig3_7_panel_{label}_{stem}_lettered_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.clf()
    plt.close(fig)
    gc.collect()


def make_combined(summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11.8, 15.2))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.02, 1.04, 0.98, 0.96],
        width_ratios=[1, 1],
        hspace=0.50,
        wspace=0.38,
    )
    axes = {
        "A": fig.add_subplot(gs[0, :]),
        "B": fig.add_subplot(gs[1, 0]),
        "C": fig.add_subplot(gs[1, 1]),
        "F": fig.add_subplot(gs[2, :]),
        "G": fig.add_subplot(gs[3, :]),
    }
    drawers = {
        "A": lambda ax: base.panel_a_evidence_timeline(ax, summary),
        "B": lambda ax: panel_b_rank_braid_clear(ax, summary),
        "C": lambda ax: base.panel_c_concordance_map(ax, summary),
        "F": lambda ax: base.panel_d_rank_residual(ax, summary),
        "G": lambda ax: base.panel_f_specificity_volume(ax, summary),
    }
    for label, ax in axes.items():
        drawers[label](ax)
        ax.text(
            -0.075,
            1.06,
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
        OUT / "Fig3_7_ABCFG_combined_clear_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        OUT / "Fig3_7_ABCFG_combined_clear_preview.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.clf()
    plt.close(fig)
    gc.collect()


def main() -> None:
    set_clear_style()
    summary, _ = load_sources()
    summary.to_csv(OUT / "Fig3_7_ABCFG_source_data.csv", index=False)

    panel_specs = [
        ("A", "multi_metric_profile", lambda ax: base.panel_a_evidence_timeline(ax, summary), (7.4, 5.0)),
        ("B", "ml_literature_rank_slope", lambda ax: panel_b_rank_braid_clear(ax, summary), (5.6, 5.2)),
        ("C", "ml_literature_priority_scatter", lambda ax: base.panel_c_concordance_map(ax, summary), (5.6, 4.9)),
        ("F", "rank_difference_lollipop", lambda ax: base.panel_d_rank_residual(ax, summary), (5.9, 4.9)),
        ("G", "direct_studies_specificity_bubble", lambda ax: base.panel_f_specificity_volume(ax, summary), (5.8, 4.8)),
    ]
    for label, stem, drawer, size in panel_specs:
        save_panel(label, stem, drawer, size)
    make_combined(summary)
    shutil.make_archive(str(ROOT / "figure_3_7_ABCFG_clear_300dpi_tiff"), "zip", OUT)
    print(f"Saved clear ABCFG panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
