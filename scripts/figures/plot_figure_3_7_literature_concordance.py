from __future__ import annotations

import json
import math
import gc
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath

from analysis_paths import CODE_ROOT, PUBMED_RESULTS


ROOT = CODE_ROOT
SRC = PUBMED_RESULTS
OUT = ROOT / "figure_3_7_literature_concordance"
OUT.mkdir(exist_ok=True)

REGION_ORDER = [
    "ACC", "NAco", "CLA", "VTA", "DMS", "CeA", "BLA", "PL", "VP",
    "OFC", "NAsh", "GP", "RSC", "CM", "IL", "MS", "LH", "AM",
]
CANDIDATES = ["ACC", "VTA", "NAco", "CLA", "BLA", "CeA", "DMS"]

COLORS = {
    "ink": "#163047",
    "muted": "#6B7785",
    "axis": "#2A2E33",
    "grid": "#DDE5EA",
    "soft": "#F4F8FA",
    "paper": "#FFFFFF",
    "ml": "#2AAE93",
    "lit": "#D3834E",
    "direct": "#3E7EA6",
    "circuit": "#C96F5A",
    "activity": "#58AF9D",
    "behavioral": "#D9A441",
    "novel": "#DD6A55",
    "convergent": "#2AAE93",
    "literature": "#4C84AE",
    "limited": "#9A8AC2",
    "neutral": "#B8C3CC",
    "band_green": "#EAF7F2",
    "band_orange": "#FFF4E7",
    "band_red": "#FCEDEB",
    "band_blue": "#EDF5FB",
}

CLASS_COLORS = {
    "convergent": COLORS["convergent"],
    "novel_candidate": COLORS["novel"],
    "literature_supported_lower_ml": COLORS["literature"],
    "limited_evidence": COLORS["limited"],
}

REGION_COLORS = {
    "ACC": "#2E7FAD",
    "VTA": "#C4863F",
    "NAco": "#22A786",
    "CLA": "#E2624B",
    "BLA": "#B97147",
    "CeA": "#A874BA",
    "DMS": "#709E42",
}


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "axes.labelweight": "regular",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_axes(ax: plt.Axes, left: bool = True, bottom: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for spine in ax.spines.values():
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(1.0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color("#111111")


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220 if ext == "png" else 600)
    fig.clf()
    plt.close(fig)
    gc.collect()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SRC / "literature_evidence_summary.csv")
    evidence = pd.read_csv(SRC / "bibliometric_evidence_type_by_region.csv")
    summary["brain_region"] = pd.Categorical(summary["brain_region"], REGION_ORDER, ordered=True)
    summary = summary.sort_values("brain_region").reset_index(drop=True)
    evidence["brain_region"] = pd.Categorical(evidence["brain_region"], REGION_ORDER, ordered=True)
    evidence = evidence.sort_values("brain_region").reset_index(drop=True)
    return summary, evidence


def load_concordance_metrics() -> dict[str, float]:
    path = SRC / "concordance_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "spearman_rho", "spearman_p", "kendall_tau", "kendall_p",
        "permutation_p", "top3_overlap", "top5_overlap",
    }
    missing = required - set(metrics)
    if missing:
        raise KeyError(f"Missing concordance metrics in {path}: {sorted(missing)}")
    return {key: float(metrics[key]) for key in required}


def norm_high_priority(rank: pd.Series) -> pd.Series:
    max_rank = float(rank.max())
    return (max_rank - rank.astype(float) + 1.0) / max_rank


def add_legend(ax: plt.Axes, handles, *, loc="upper center", anchor=(0.5, 1.08), ncol=3) -> None:
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, frameon=False,
              ncol=ncol, handlelength=1.0, handletextpad=0.35, columnspacing=0.85)


def panel_a_evidence_timeline(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s.copy()
    metrics = [
        ("Direct", np.log10(df["direct_original_studies"].astype(float) + 1.0)),
        ("Intervention/\ncircuit", np.log10(df["intervention_or_circuit_studies_auto"].astype(float) + 1.0)),
        ("Specificity", df["specificity_ratio"].astype(float)),
        ("Co-occurrence", df["normalized_cooccurrence_strength"].astype(float)),
        ("ML stability", df["stability"].astype(float)),
    ]
    xs = np.arange(len(metrics))
    values = []
    for _, series in metrics:
        arr = series.to_numpy(dtype=float)
        span = np.nanmax(arr) - np.nanmin(arr)
        values.append((arr - np.nanmin(arr)) / span if span > 0 else np.zeros_like(arr))
    mat = np.vstack(values).T
    for x, (label, _) in zip(xs, metrics):
        ax.axvline(x, color=COLORS["grid"], linewidth=0.85, zorder=0)
        ax.text(x, 1.075, label, ha="center", va="bottom", fontsize=7.1,
                fontweight="bold", color=COLORS["ink"])
    left_label_y = {
        "VTA": 1.03,
        "ACC": 0.91,
        "NAco": 0.84,
        "BLA": 0.73,
        "CeA": 0.64,
        "DMS": 0.58,
        "CLA": 0.45,
    }
    for i, row in df.iterrows():
        region = str(row["brain_region"])
        color = REGION_COLORS.get(region, CLASS_COLORS.get(row["concordance_class"], COLORS["neutral"]))
        highlighted = region in CANDIDATES or region in ["NAsh", "PL", "OFC"]
        ax.plot(xs, mat[i], color=color, linewidth=2.0 if highlighted else 0.9,
                alpha=0.88 if highlighted else 0.22, zorder=2)
        ax.scatter(xs, mat[i], s=30 if highlighted else 11, color=color,
                   alpha=0.90 if highlighted else 0.25, edgecolor="white",
                   linewidth=0.45, zorder=3)
        if region in CANDIDATES:
            ly = left_label_y.get(region, mat[i, 0])
            ax.plot([-0.08, -0.20], [mat[i, 0], ly], color=color,
                    linewidth=0.55, alpha=0.70, zorder=2)
            ax.text(-0.24, ly, region, ha="right", va="center",
                    fontsize=7.2, fontweight="bold", color=color)
    ax.set_xlim(-0.62, len(metrics) - 0.30)
    ax.set_ylim(-0.04, 1.15)
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Direction-aligned normalized value")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    handles = [
        Line2D([0], [0], color=COLORS["convergent"], label="Convergent", linewidth=1.8),
        Line2D([0], [0], color=COLORS["novel"], label="Data-driven candidate", linewidth=1.8),
        Line2D([0], [0], color=COLORS["literature"], label="Literature-supported", linewidth=1.8),
        Line2D([0], [0], color=COLORS["limited"], label="Limited", linewidth=1.8),
    ]
    add_legend(ax, handles, anchor=(0.50, 1.17), ncol=4)
    style_axes(ax, bottom=False)


def panel_b_rank_braid(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s.copy()
    x_ml, x_lit = 0.0, 1.0
    for _, row in df.iterrows():
        region = str(row["brain_region"])
        color = REGION_COLORS.get(region, CLASS_COLORS.get(row["concordance_class"], COLORS["neutral"]))
        lw = 2.1 if region in CANDIDATES else 1.1
        alpha = 0.95 if region in CANDIDATES else 0.36
        ax.plot([x_ml, x_lit], [row["ml_rank"], row["literature_composite_rank"]],
                color=color, linewidth=lw, alpha=alpha, zorder=2)
        ax.scatter([x_ml, x_lit], [row["ml_rank"], row["literature_composite_rank"]],
                   s=45 if region in CANDIDATES else 24, color=color,
                   edgecolor="white", linewidth=0.7, zorder=3)
        if region in CANDIDATES:
            ax.text(x_ml - 0.055, row["ml_rank"], region, ha="right", va="center",
                    fontsize=7.5, fontweight="bold", color=color)
            ax.text(x_lit + 0.055, row["literature_composite_rank"], region,
                    ha="left", va="center", fontsize=7.5, fontweight="bold", color=color)
    for x, lab in [(x_ml, "ML rank"), (x_lit, "Literature rank")]:
        ax.axvline(x, color=COLORS["axis"], linewidth=1.0)
        ax.text(x, 0.05, lab, ha="center", va="bottom", fontsize=8.0,
                fontweight="bold", color=COLORS["ink"])
    ax.axhspan(0.5, 3.5, color=COLORS["band_green"], alpha=0.70, zorder=0)
    ax.axhspan(3.5, 7.5, color=COLORS["band_orange"], alpha=0.38, zorder=0)
    ax.set_xlim(-0.34, 1.34)
    ax.set_ylim(18.7, 0.35)
    ax.set_yticks([1, 3, 5, 10, 15, 18])
    ax.set_ylabel("Rank, lower is higher priority")
    ax.set_xticks([])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    style_axes(ax, bottom=False)


def panel_c_concordance_map(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s.copy()
    df["ml_priority"] = norm_high_priority(df["ml_rank"])
    df["lit_priority"] = norm_high_priority(df["literature_composite_rank"])
    max_direct = max(df["direct_original_studies"].max(), 1)
    ax.axvspan(0.5, 1.02, ymin=0.5, ymax=1.0, color=COLORS["band_green"], alpha=0.55, zorder=0)
    ax.axvspan(0.5, 1.02, ymin=0.0, ymax=0.5, color=COLORS["band_red"], alpha=0.45, zorder=0)
    ax.axvspan(-0.02, 0.5, ymin=0.5, ymax=1.0, color=COLORS["band_blue"], alpha=0.50, zorder=0)
    ax.axvspan(-0.02, 0.5, ymin=0.0, ymax=0.5, color=COLORS["soft"], alpha=0.80, zorder=0)
    ax.axhline(0.5, color=COLORS["axis"], linewidth=0.75, alpha=0.65)
    ax.axvline(0.5, color=COLORS["axis"], linewidth=0.75, alpha=0.65)
    label_offsets = {
        "ACC": (-0.052, -0.045, "right"),
        "VTA": (0.025, -0.055, "left"),
        "NAco": (-0.045, -0.025, "right"),
        "CLA": (-0.030, -0.070, "right"),
        "BLA": (0.038, -0.035, "left"),
        "CeA": (0.040, -0.055, "left"),
        "DMS": (0.050, 0.060, "left"),
        "NAsh": (0.026, 0.050, "left"),
        "OFC": (0.027, 0.040, "left"),
        "PL": (0.026, 0.038, "left"),
    }
    for _, row in df.iterrows():
        region = str(row["brain_region"])
        color = REGION_COLORS.get(region, CLASS_COLORS.get(row["concordance_class"], COLORS["neutral"]))
        size = 26 + 245 * (row["direct_original_studies"] / max_direct)
        ax.scatter(row["ml_priority"], row["lit_priority"], s=size, color=color,
                   alpha=0.86 if region in CANDIDATES else 0.38,
                   edgecolor="white", linewidth=0.8, zorder=3)
        if region in CANDIDATES or region in ["NAsh", "OFC", "PL"]:
            dx, dy, ha = label_offsets.get(region, (0.03, 0.03, "left"))
            lx = np.clip(row["ml_priority"] + dx, 0.02, 0.99)
            ly = np.clip(row["lit_priority"] + dy, 0.03, 0.99)
            ax.plot([row["ml_priority"], lx], [row["lit_priority"], ly],
                    color=color, linewidth=0.55, alpha=0.65, zorder=2)
            ax.text(lx, ly, region, ha=ha, va="center", fontsize=7.1, fontweight="bold",
                    color=color)
    ax.text(0.76, 0.04, "Data-driven\ncandidate", ha="center", va="bottom",
            fontsize=7.2, fontweight="bold", color=COLORS["novel"])
    ax.text(0.25, 0.97, "Literature-\nsupported", ha="center", va="top",
            fontsize=7.2, fontweight="bold", color=COLORS["literature"])
    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("Machine-learning priority")
    ax.set_ylabel("Composite literature priority")
    ax.grid(color=COLORS["grid"], linewidth=0.42, alpha=0.50)
    style_axes(ax)


def panel_d_rank_residual(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s.copy()
    df["rank_residual"] = df["literature_composite_rank"] - df["ml_rank"]
    df = df.sort_values("rank_residual").reset_index(drop=True)
    y = np.arange(len(df))
    ax.axvline(0, color=COLORS["axis"], linewidth=0.9)
    ax.axvspan(-12, 0, color=COLORS["band_blue"], alpha=0.45, zorder=0)
    ax.axvspan(0, 16, color=COLORS["band_red"], alpha=0.38, zorder=0)
    for yi, row in zip(y, df.itertuples()):
        color = REGION_COLORS.get(row.brain_region, CLASS_COLORS.get(row.concordance_class, COLORS["neutral"]))
        ax.plot([0, row.rank_residual], [yi, yi], color=color, linewidth=1.8,
                alpha=0.88 if row.brain_region in CANDIDATES else 0.42,
                solid_capstyle="round")
        ax.scatter(row.rank_residual, yi, s=45 if row.brain_region in CANDIDATES else 24,
                   color=color, edgecolor="white", linewidth=0.7, zorder=3)
        if row.brain_region in CANDIDATES:
            offset = 0.38 if row.rank_residual >= 0 else -0.38
            ax.text(row.rank_residual + offset, yi, f"{int(row.rank_residual):+d}",
                    ha="left" if offset > 0 else "right", va="center",
                    fontsize=7.0, fontweight="bold", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(df["brain_region"].astype(str))
    ax.set_xlabel("Literature rank minus ML rank")
    ax.set_xlim(-10.5, 13.8)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    style_axes(ax)


def donut(ax: plt.Axes, center: tuple[float, float], values: list[float],
          colors: list[str], radius: float = 0.38, width: float = 0.095) -> None:
    total = sum(values)
    start = 90.0
    if total <= 0:
        ax.add_patch(Wedge(center, radius, 0, 360, width=width,
                           facecolor="#EEF2F5", edgecolor="white", linewidth=0.7))
        return
    for val, color in zip(values, colors):
        theta = 360.0 * val / total
        ax.add_patch(Wedge(center, radius, start, start - theta, width=width,
                           facecolor=color, edgecolor="white", linewidth=0.7))
        start -= theta


def panel_e_evidence_glyphs(ax: plt.Axes, s: pd.DataFrame, evidence: pd.DataFrame) -> None:
    ev = evidence.set_index("brain_region")
    evidence_cols = [
        ("Intervention", "intervention", COLORS["circuit"]),
        ("Circuit", "circuit", COLORS["direct"]),
        ("Activity", "activity", COLORS["activity"]),
        ("Behavioral", "behavioral", COLORS["behavioral"]),
    ]
    left_y = np.linspace(0.88, 0.12, len(CANDIDATES))
    right_y = np.linspace(0.78, 0.22, len(evidence_cols))
    x0, x1 = 0.10, 0.88
    max_value = max(float(evidence[[col for _, col, _ in evidence_cols]].max().max()), 1.0)
    for yi, region in zip(left_y, CANDIDATES):
        rcolor = REGION_COLORS.get(region, COLORS["ink"])
        ax.scatter(x0, yi, s=42, color=rcolor, edgecolor="white", linewidth=0.75, zorder=4)
        ax.text(x0 - 0.035, yi, region, ha="right", va="center",
                fontsize=7.8, fontweight="bold", color=rcolor)
    for yi, (label, _, color) in zip(right_y, evidence_cols):
        ax.scatter(x1, yi, s=54, color=color, edgecolor="white", linewidth=0.75, zorder=4)
        ax.text(x1 + 0.035, yi, label, ha="left", va="center",
                fontsize=7.2, fontweight="bold", color=color)
    for ry, region in zip(left_y, CANDIDATES):
        for ty, (_, col, color) in zip(right_y, evidence_cols):
            value = float(ev.loc[region, col]) if region in ev.index else 0.0
            if value <= 0:
                continue
            verts = [
                (x0 + 0.025, ry),
                (0.34, ry),
                (0.62, ty),
                (x1 - 0.025, ty),
            ]
            path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            ax.add_patch(PathPatch(
                path, facecolor="none", edgecolor=color,
                linewidth=0.45 + 4.8 * math.sqrt(value / max_value),
                alpha=0.18 + 0.38 * math.sqrt(value / max_value),
                capstyle="round", joinstyle="round", zorder=1))
    ax.plot([x0, x0], [0.06, 0.94], color=COLORS["grid"], linewidth=1.0, zorder=0)
    ax.plot([x1, x1], [0.12, 0.88], color=COLORS["grid"], linewidth=1.0, zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def panel_f_specificity_volume(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s.copy()
    x = np.log10(df["direct_original_studies"].astype(float) + 1.0)
    y = df["specificity_ratio"] * 100.0
    cooc = df["normalized_cooccurrence_strength"]
    max_cooc = max(cooc.max(), 1e-9)
    label_offsets = {
        "NAco": (0.14, 0.34),
        "VTA": (0.08, 0.18),
        "ACC": (0.07, -0.10),
        "CLA": (0.06, 0.13),
        "BLA": (0.08, -0.12),
        "CeA": (0.09, 0.12),
        "DMS": (0.06, 0.24),
        "NAsh": (0.08, 0.20),
        "PL": (0.06, 0.18),
        "IL": (0.06, 0.18),
    }
    for _, row in df.iterrows():
        region = str(row["brain_region"])
        color = REGION_COLORS.get(region, CLASS_COLORS.get(row["concordance_class"], COLORS["neutral"]))
        xi = math.log10(float(row["direct_original_studies"]) + 1.0)
        yi = float(row["specificity_ratio"]) * 100.0
        size = 34 + 270 * (float(row["normalized_cooccurrence_strength"]) / max_cooc)
        ax.scatter(xi, yi, s=size, color=color,
                   alpha=0.88 if region in CANDIDATES else 0.34,
                   edgecolor="white", linewidth=0.8, zorder=3)
        if region in CANDIDATES or region in ["NAsh", "PL", "IL"]:
            dx, dy = label_offsets.get(region, (0.06, 0.10))
            lx, ly = xi + dx, yi + dy
            ax.plot([xi, lx], [yi, ly], color=color, linewidth=0.5, alpha=0.65, zorder=2)
            ax.text(lx, ly, region, ha="left", va="center",
                    fontsize=7.0, fontweight="bold", color=color)
    ax.set_xlabel("Direct studies, log10(count + 1)")
    ax.set_ylabel("Literature specificity (%)")
    ax.set_xlim(-0.05, max(x) + 0.22)
    ax.set_ylim(-0.15, max(y) + 0.75)
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.62)
    style_axes(ax)


def panel_g_statistics(ax: plt.Axes, metrics: dict[str, float]) -> None:
    rng = np.random.default_rng(3707)
    n = 18
    null_rho = np.clip(rng.normal(0, 1 / math.sqrt(n - 1), 10000), -1, 1)
    null_tau = np.clip(rng.normal(0, math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1))), 10000), -1, 1)
    rows = [
        ("Spearman rho", null_rho, metrics["spearman_rho"],
         f"P = {metrics['spearman_p']:.4f}", COLORS["ml"], 1.0),
        ("Kendall tau", null_tau, metrics["kendall_tau"],
         f"P = {metrics['kendall_p']:.4f}", COLORS["lit"], 0.0),
    ]
    for label, vals, obs, ptxt, color, base in rows:
        hist, edges = np.histogram(vals, bins=70, range=(-0.85, 0.85), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        hist = hist / hist.max() * 0.34
        ax.fill_between(centers, base, base + hist, color=color, alpha=0.24, linewidth=0)
        ax.plot(centers, base + hist, color=color, linewidth=1.2)
        ax.axvline(obs, ymin=(base - 0.02) / 1.45, ymax=(base + 0.40) / 1.45,
                   color=color, linewidth=1.6)
        ax.scatter(obs, base + 0.38, marker="v", s=38, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(-0.82, base + 0.20, label, ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=COLORS["ink"])
        ax.text(obs + 0.035, base + 0.38, f"{obs:.3f}", ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=color)
        ax.text(0.82, base + 0.20, ptxt, ha="right", va="center",
                fontsize=6.8, fontweight="bold", color=color)
    ax.text(-0.82, -0.30, "Top-k overlap", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=COLORS["ink"])
    overlap_rows = [
        (-0.25, metrics["top3_overlap"], "Top-3", COLORS["direct"]),
        (0.36, metrics["top5_overlap"], "Top-5", COLORS["limited"]),
    ]
    for x0, val, name, color in overlap_rows:
        label = f"{name} {100 * val:.1f}%"
        ax.add_patch(Rectangle((x0, -0.42), 0.46, 0.16, facecolor="#E8EEF2", edgecolor="none"))
        ax.add_patch(Rectangle((x0, -0.42), 0.46 * val, 0.16, facecolor=color, edgecolor="none", alpha=0.86))
        ax.text(x0 + 0.23, -0.16, label, ha="center", va="center",
                fontsize=6.7, fontweight="bold", color=color)
    ax.set_xlim(-0.86, 0.86)
    ax.set_ylim(-0.52, 1.42)
    ax.set_xlabel("Null concordance coefficient")
    ax.set_yticks([])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    style_axes(ax, left=False)


def panel_h_candidate_scoreboard(ax: plt.Axes, s: pd.DataFrame) -> None:
    df = s[s["brain_region"].astype(str).isin(CANDIDATES)].copy()
    df["brain_region"] = pd.Categorical(df["brain_region"].astype(str), CANDIDATES, ordered=True)
    df = df.sort_values("brain_region")
    verts = {
        "ML priority": np.array([0.06, 0.08]),
        "Literature priority": np.array([0.94, 0.08]),
        "Intervention/circuit": np.array([0.50, 0.96]),
    }
    tri = np.vstack([verts["ML priority"], verts["Literature priority"],
                     verts["Intervention/circuit"], verts["ML priority"]])
    ax.add_patch(PathPatch(
        MplPath([tuple(verts["ML priority"]), (0.50, 0.08), (0.37, 0.34), tuple(verts["ML priority"])],
                [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]),
        facecolor=COLORS["band_green"], edgecolor="none", alpha=0.48, zorder=0))
    ax.add_patch(PathPatch(
        MplPath([tuple(verts["Literature priority"]), (0.50, 0.08), (0.63, 0.34), tuple(verts["Literature priority"])],
                [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]),
        facecolor=COLORS["band_orange"], edgecolor="none", alpha=0.45, zorder=0))
    ax.add_patch(PathPatch(
        MplPath([tuple(verts["Intervention/circuit"]), (0.37, 0.34), (0.63, 0.34), tuple(verts["Intervention/circuit"])],
                [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY]),
        facecolor=COLORS["band_blue"], edgecolor="none", alpha=0.38, zorder=0))
    ax.plot(tri[:, 0], tri[:, 1], color=COLORS["axis"], linewidth=1.0)
    for frac in [0.20, 0.40, 0.60, 0.80]:
        p1 = verts["ML priority"] * frac + verts["Intervention/circuit"] * (1 - frac)
        p2 = verts["Literature priority"] * frac + verts["Intervention/circuit"] * (1 - frac)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COLORS["grid"], linewidth=0.55, alpha=0.80)
        p1 = verts["ML priority"] * frac + verts["Literature priority"] * (1 - frac)
        p2 = verts["Intervention/circuit"] * frac + verts["Literature priority"] * (1 - frac)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COLORS["grid"], linewidth=0.55, alpha=0.80)
        p1 = verts["Literature priority"] * frac + verts["ML priority"] * (1 - frac)
        p2 = verts["Intervention/circuit"] * frac + verts["ML priority"] * (1 - frac)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=COLORS["grid"], linewidth=0.55, alpha=0.80)
    tick_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.10}
    for value in [0.25, 0.50, 0.75]:
        pos = verts["Intervention/circuit"] * (1 - value) + verts["ML priority"] * value
        ax.text(pos[0] + 0.020, pos[1] + 0.004, f"{value:.2f}",
                ha="left", va="center", fontsize=5.4, color=COLORS["muted"],
                alpha=0.88, bbox=tick_box)
        pos = verts["ML priority"] * (1 - value) + verts["Literature priority"] * value
        ax.text(pos[0], pos[1] + 0.025, f"{value:.2f}",
                ha="center", va="center", fontsize=5.4, color=COLORS["muted"],
                alpha=0.88, bbox=tick_box)
        pos = verts["Literature priority"] * (1 - value) + verts["Intervention/circuit"] * value
        ax.text(pos[0] - 0.016, pos[1], f"{value:.2f}",
                ha="right", va="center", fontsize=5.4, color=COLORS["muted"],
                alpha=0.88, bbox=tick_box)
    label_positions = {
        "VTA": (0.115, 0.69, "right"),
        "DMS": (0.115, 0.58, "right"),
        "CeA": (0.115, 0.47, "right"),
        "BLA": (0.115, 0.36, "right"),
        "NAco": (0.905, 0.65, "left"),
        "CLA": (0.920, 0.51, "left"),
        "ACC": (0.905, 0.30, "left"),
    }
    leader_curve = {
        "VTA": 0.18,
        "DMS": 0.12,
        "CeA": 0.06,
        "BLA": -0.05,
        "NAco": -0.15,
        "CLA": -0.08,
        "ACC": 0.04,
    }
    max_direct = max(float(df["direct_original_studies"].max()), 1.0)
    point_rows = []
    for _, row in df.iterrows():
        region = str(row["brain_region"])
        color = REGION_COLORS.get(region, COLORS["ink"])
        ml = (18.0 - float(row["ml_rank"]) + 1.0) / 18.0
        lit = (18.0 - float(row["literature_composite_rank"]) + 1.0) / 18.0
        ic = float(row["intervention_or_circuit_studies_auto"]) / max(float(row["direct_original_studies"]), 1.0)
        weights = np.array([ml, lit, ic], dtype=float)
        weights = weights / weights.sum()
        xy = (weights[0] * verts["ML priority"] +
              weights[1] * verts["Literature priority"] +
              weights[2] * verts["Intervention/circuit"])
        point_rows.append((row, region, color, xy))

    def hull(points: list[np.ndarray]) -> list[np.ndarray]:
        pts = sorted({(float(p[0]), float(p[1])) for p in points})
        if len(pts) <= 2:
            return [np.array(p) for p in pts]
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return [np.array(p) for p in lower[:-1] + upper[:-1]]

    hull_pts = hull([xy for _, _, _, xy in point_rows])
    if len(hull_pts) >= 3:
        closed = hull_pts + [hull_pts[0]]
        ax.add_patch(PathPatch(
            MplPath([tuple(p) for p in closed],
                    [MplPath.MOVETO] + [MplPath.LINETO] * (len(closed) - 2) + [MplPath.CLOSEPOLY]),
            facecolor="#EEF5F6", edgecolor=COLORS["ink"], linewidth=0.75,
            alpha=0.26, zorder=2))

    for row, region, color, xy in point_rows:
        size = 50 + 210 * math.sqrt(float(row["direct_original_studies"]) / max_direct)
        ax.scatter(xy[0], xy[1], s=size, color=color, alpha=0.82,
                   edgecolor="white", linewidth=1.0, zorder=4)
        ic_ratio = (float(row["intervention_or_circuit_studies_auto"]) /
                    max(float(row["direct_original_studies"]), 1.0))
        ax.scatter(xy[0], xy[1], s=size * max(0.10, min(ic_ratio, 1.0)),
                   facecolor="none", edgecolor=color, linewidth=1.4, zorder=5)
        lx, ly, ha = label_positions.get(region, (xy[0] + 0.04, xy[1] + 0.04, "left"))
        ax.annotate(
            region,
            xy=(xy[0], xy[1]),
            xytext=(lx, ly),
            ha=ha,
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=color,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 0.22},
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "lw": 0.55,
                "alpha": 0.54,
                "shrinkA": 3,
                "shrinkB": 7,
                "connectionstyle": f"arc3,rad={leader_curve.get(region, 0.0)}",
            },
        )
        direct_frac = float(row["direct_original_studies"]) / max_direct
        bar_len = 0.052 * math.sqrt(direct_frac)
        ic_bar = bar_len * min(ic_ratio, 1.0)
        ybar = ly - 0.038
        if ha == "right":
            ax.plot([lx - bar_len, lx], [ybar, ybar], color="#CFD9DF",
                    linewidth=2.2, solid_capstyle="round", zorder=6)
            ax.plot([lx - ic_bar, lx], [ybar, ybar], color=color,
                    linewidth=2.2, solid_capstyle="round", zorder=7)
        else:
            ax.plot([lx, lx + bar_len], [ybar, ybar], color="#CFD9DF",
                    linewidth=2.2, solid_capstyle="round", zorder=6)
            ax.plot([lx, lx + ic_bar], [ybar, ybar], color=color,
                    linewidth=2.2, solid_capstyle="round", zorder=7)
    ax.text(0.06, 0.025, "ML priority\nnormalized", ha="left", va="top",
            fontsize=6.4, fontweight="bold", color=COLORS["ml"], linespacing=0.95)
    ax.text(0.94, 0.025, "Literature evidence\nnormalized", ha="right", va="top",
            fontsize=6.4, fontweight="bold", color=COLORS["lit"], linespacing=0.95)
    ax.text(0.50, 1.015, "Intervention/circuit\nshare", ha="center", va="bottom",
            fontsize=6.4, fontweight="bold", color=COLORS["direct"], linespacing=0.95)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.015, 1.045)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def make_figures() -> None:
    s, evidence = load_data()
    metrics = load_concordance_metrics()
    s.to_csv(OUT / "figure_3_7_literature_evidence_source.csv", index=False)
    evidence.to_csv(OUT / "figure_3_7_evidence_type_source.csv", index=False)
    (OUT / "figure_3_7_statistics_used.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    panels = [
        ("premium_panel_A_evidence_timeline", lambda ax: panel_a_evidence_timeline(ax, s), (6.1, 4.7)),
        ("premium_panel_B_rank_braid", lambda ax: panel_b_rank_braid(ax, s), (4.2, 4.7)),
        ("premium_panel_C_concordance_map", lambda ax: panel_c_concordance_map(ax, s), (4.6, 4.2)),
        ("premium_panel_D_rank_residual", lambda ax: panel_d_rank_residual(ax, s), (4.9, 4.5)),
        ("premium_panel_E_evidence_glyphs", lambda ax: panel_e_evidence_glyphs(ax, s, evidence), (6.0, 2.5)),
        ("premium_panel_F_specificity_volume", lambda ax: panel_f_specificity_volume(ax, s), (4.8, 4.0)),
        ("premium_panel_G_concordance_statistics", lambda ax: panel_g_statistics(ax, metrics), (5.7, 2.1)),
        ("premium_panel_H_candidate_scoreboard", lambda ax: panel_h_candidate_scoreboard(ax, s), (6.6, 3.9)),
    ]
    for stem, drawer, size in panels:
        fig, ax = plt.subplots(figsize=size)
        drawer(ax)
        save_all(fig, stem)

    fig = plt.figure(figsize=(14.2, 16.2))
    gs = fig.add_gridspec(
        4, 6,
        height_ratios=[1.16, 1.00, 0.66, 2.35],
        width_ratios=[1, 1, 1, 1, 1, 1],
        wspace=0.58,
        hspace=0.56,
    )
    axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
        fig.add_subplot(gs[2, 0:3]),
        fig.add_subplot(gs[2, 3:6]),
        fig.add_subplot(gs[3, 0:6]),
    ]
    drawers = [
        lambda ax: panel_a_evidence_timeline(ax, s),
        lambda ax: panel_b_rank_braid(ax, s),
        lambda ax: panel_c_concordance_map(ax, s),
        lambda ax: panel_d_rank_residual(ax, s),
        lambda ax: panel_f_specificity_volume(ax, s),
        lambda ax: panel_e_evidence_glyphs(ax, s, evidence),
        lambda ax: panel_g_statistics(ax, metrics),
        lambda ax: panel_h_candidate_scoreboard(ax, s),
    ]
    for ax, drawer in zip(axes, drawers):
        drawer(ax)
    for ext in ("png", "pdf", "svg"):
        if ext == "png":
            fig.savefig(OUT / f"figure_3_7_literature_concordance_combined.{ext}",
                        dpi=220)
        else:
            fig.savefig(OUT / f"figure_3_7_literature_concordance_combined.{ext}",
                        dpi=600, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    gc.collect()


def main() -> None:
    set_style()
    make_figures()
    shutil.make_archive(str(ROOT / "figure_3_7_literature_concordance_package"), "zip", OUT)
    print(f"Saved Figure 3.7 literature concordance panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
