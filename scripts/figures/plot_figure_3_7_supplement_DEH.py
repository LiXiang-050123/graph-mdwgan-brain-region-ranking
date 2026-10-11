from __future__ import annotations

import gc
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import plot_figure_3_7_literature_concordance as base
from plot_figure_3_7_ABCFG_clear import load_sources
from analysis_paths import CODE_ROOT


ROOT = CODE_ROOT
OUT = ROOT / "figure_3_7_supplement_DEH_300dpi_tiff"
OUT.mkdir(exist_ok=True)


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


def save_panel(label: str, stem: str, drawer, size: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=size)
    drawer(ax)
    fig.savefig(
        OUT / f"Fig3_7_supplement_panel_{label}_{stem}_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        OUT / f"Fig3_7_supplement_panel_{label}_{stem}_preview.png",
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
        OUT / f"Fig3_7_supplement_panel_{label}_{stem}_lettered_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.clf()
    plt.close(fig)
    gc.collect()


def make_combined(summary, evidence, metrics) -> None:
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.72, 1.0],
        width_ratios=[1.06, 1.0],
        hspace=0.54,
        wspace=0.34,
    )
    axes = [
        fig.add_subplot(gs[0, :]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    drawers = [
        lambda ax: base.panel_e_evidence_glyphs(ax, summary, evidence),
        lambda ax: base.panel_g_statistics(ax, metrics),
        lambda ax: base.panel_h_candidate_scoreboard(ax, summary),
    ]
    for label, ax, drawer in zip(["A", "B", "C"], axes, drawers):
        drawer(ax)
        ax.text(
            -0.065,
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
        OUT / "Fig3_7_supplement_DEH_combined_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        OUT / "Fig3_7_supplement_DEH_combined_preview.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.clf()
    plt.close(fig)
    gc.collect()


def main() -> None:
    set_clear_style()
    summary, evidence = load_sources()
    metrics = base.load_concordance_metrics()
    summary.to_csv(OUT / "Fig3_7_supplement_summary_source_data.csv", index=False)
    evidence.to_csv(OUT / "Fig3_7_supplement_evidence_type_source_data.csv", index=False)

    panel_specs = [
        ("A", "evidence_type_flow", lambda ax: base.panel_e_evidence_glyphs(ax, summary, evidence), (7.2, 3.0)),
        ("B", "concordance_statistics", lambda ax: base.panel_g_statistics(ax, metrics), (6.3, 2.8)),
        ("C", "candidate_evidence_scoreboard", lambda ax: base.panel_h_candidate_scoreboard(ax, summary), (6.7, 4.4)),
    ]
    for label, stem, drawer, size in panel_specs:
        save_panel(label, stem, drawer, size)
    make_combined(summary, evidence, metrics)
    shutil.make_archive(str(ROOT / "figure_3_7_supplement_DEH_300dpi_tiff"), "zip", OUT)
    print(f"Saved Figure 3.7 supplementary panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
