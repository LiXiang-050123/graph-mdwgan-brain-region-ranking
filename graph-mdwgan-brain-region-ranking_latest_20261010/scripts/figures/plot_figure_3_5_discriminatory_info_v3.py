"""
Alternative Q1-style Figure 3.5 set.

This version follows the user's reference figures more closely:
- effect-size scatter similar to reference panel A
- faceted sample-plane scatter similar to reference panel C
- violin/box distributions for fold-level performance
- forest/lollipop/curve panels as supporting evidence
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import gaussian_kde

from analysis_paths import CODE_ROOT, CORRECTED_SECONDARY_RESULTS, PRIMARY_CV_RESULTS


ROOT = CODE_ROOT
CV = PRIMARY_CV_RESULTS
ONEZONE = CORRECTED_SECONDARY_RESULTS
OUT = ROOT / "figure_3_5_discriminatory_info_v3"
OUT.mkdir(exist_ok=True)

COLORS = {
    "ink": "#21364D",
    "teal": "#2B9A92",
    "teal_light": "#9BD7D0",
    "coral": "#D86B61",
    "coral_light": "#EFB7B1",
    "blue": "#4D93C7",
    "gold": "#C79E47",
    "plum": "#8D70B3",
    "green": "#58B487",
    "gray": "#9BA5AF",
    "pale": "#F5F7F9",
    "line": "#DEE4EA",
    "axis": "#2E3338",
}

SCHEME_COLORS = {"consensus": COLORS["teal"], "fusion": COLORS["coral"]}
REGION_COLORS = {
    "ACC": COLORS["teal"],
    "VTA": COLORS["coral"],
    "NAco": COLORS["blue"],
    "CLA": COLORS["gold"],
    "BLA": COLORS["plum"],
    "CeA": COLORS["green"],
    "DMS": "#708093",
    "VP": "#B88B56",
    "PL": COLORS["gray"],
}


def set_style() -> None:
    plt.rcParams.update(
        {
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
        }
    )


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


def style_profile_panel(ax: plt.Axes, *, xlabel: str, ylabel: str, ylim: tuple[float, float],
                        yticks: list[float] | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 25)
    ax.set_ylim(*ylim)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.55, alpha=0.72)
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.30, alpha=0.24)
    style_axes(ax)


def save_all(fig: plt.Figure, stem: str, *, bbox_inches: str | None = "tight") -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, bbox_inches=bbox_inches)
    fig.savefig(
        OUT / f"{stem}_300dpi.tif",
        dpi=300,
        bbox_inches=bbox_inches,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def perf_summary() -> pd.DataFrame:
    df = pd.read_csv(CV / "auc_k_mean_sd.csv")
    return df[(df["method"].eq("Ours")) & (df["scheme"].isin(["consensus", "fusion", "gnn"]))].copy()


def perf_long() -> pd.DataFrame:
    df = pd.read_csv(CV / "auc_k_long.csv")
    return df[(df["method"].eq("Ours")) & (df["scheme"].isin(["consensus", "fusion", "gnn"]))].copy()


def bootstrap_mean_ci(values: np.ndarray, seed: int = 1, n_boot: int = 6000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def selected_features_file(fold: int = 1, k: int = 3) -> pd.DataFrame:
    return pd.read_csv(CV / f"fold_{fold:04d}" / "fs_OURS" / f"cv_top{k}.csv")


def top3_records(scheme: str = "fusion") -> list[tuple[int, tuple[str, ...]]]:
    records: list[tuple[int, tuple[str, ...]]] = []
    for fold_dir in sorted(CV.glob("fold_*/fs_OURS")):
        fold = int(fold_dir.parent.name.split("_")[-1])
        if scheme == "fusion":
            df = pd.read_csv(fold_dir / "gnn_results" / "consensus_plus_gnn.csv")
            regs = tuple(df.sort_values("ConsensusPlusGNN", ascending=False).head(3)["BrainRegion"].tolist())
        else:
            df = pd.read_csv(fold_dir / "consensus_ranking.csv")
            regs = tuple(df.sort_values("avg_rank", ascending=True).head(3)["BrainRegion"].tolist())
        records.append((fold, regs))
    return records


def kde_violin(ax: plt.Axes, values: np.ndarray, center: float, color: str, width: float = 0.34) -> None:
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) < 2:
        yy = np.linspace(values[0] - 0.012, values[0] + 0.012, 50)
        dens = np.sin(np.linspace(0, np.pi, len(yy))) * width
    else:
        jittered = values + np.random.default_rng(99).normal(0, 0.0025, len(values))
        yy = np.linspace(max(0, values.min() - 0.05), min(1.02, values.max() + 0.025), 220)
        dens = gaussian_kde(jittered, bw_method=0.18)(yy)
        dens = dens / dens.max() * width
    ax.fill_betweenx(yy, center - dens, center + dens, color=color, alpha=0.22, linewidth=0)
    ax.plot(center - dens, yy, color=color, linewidth=1.05)
    ax.plot(center + dens, yy, color=color, linewidth=1.05)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    ax.add_patch(Rectangle((center - width * 0.36, q1), width * 0.72, q3 - q1,
                           facecolor="white", edgecolor=COLORS["axis"], linewidth=0.75, zorder=3))
    ax.plot([center - width * 0.36, center + width * 0.36], [med, med],
            color=COLORS["axis"], linewidth=1.1, zorder=4)


def panel_a_lowdim_embedding(ax: plt.Axes) -> None:
    df = selected_features_file(1, 3).copy()
    features = ["VTA", "ACC", "NAco"]
    xmat = df[features].to_numpy(dtype=float)
    z = (xmat - xmat.mean(axis=0)) / xmat.std(axis=0, ddof=1)
    y = df["Group"].map({"Saline": 0, "METH": 1}).to_numpy()
    mean0 = z[y == 0].mean(axis=0)
    mean1 = z[y == 1].mean(axis=0)
    cov = np.cov(z.T) + np.eye(z.shape[1]) * 1e-4
    w = np.linalg.pinv(cov) @ (mean1 - mean0)
    w = w / np.linalg.norm(w)
    residual = z - np.outer(z @ w, w)
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    v = vt[0]
    if abs(np.dot(v, w)) > 1e-6:
        v = v - np.dot(v, w) * w
        v = v / np.linalg.norm(v)
    coords = np.column_stack([z @ w, z @ v])
    loadings = np.column_stack([w, v])
    xmin, xmax = coords[:, 0].min() - 0.45, coords[:, 0].max() + 0.45
    ymin, ymax = coords[:, 1].min() - 0.45, coords[:, 1].max() + 0.45
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 150), np.linspace(ymin, ymax, 150))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    for group, color in [("Saline", COLORS["gray"]), ("METH", COLORS["teal"])]:
        pts = coords[df["Group"].eq(group)]
        if len(pts) > 3:
            try:
                zz = gaussian_kde(pts.T, bw_method=0.42)(grid).reshape(xx.shape)
                levels = np.linspace(zz.max() * 0.18, zz.max() * 0.94, 6)
                ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=0.035, zorder=0)
                ax.contour(xx, yy, zz, levels=levels[1::2], colors=[color], alpha=0.22,
                           linewidths=0.55, zorder=1)
            except np.linalg.LinAlgError:
                pass
    centroids = {}
    for group, color in [("Saline", COLORS["gray"]), ("METH", COLORS["teal"])]:
        for origin, marker, alpha, size in [("RealTrain", "o", 0.88, 34), ("Synthetic", "o", 0.22, 15)]:
            mask = df["Group"].eq(group) & df["Origin"].eq(origin)
            if mask.any():
                ax.scatter(coords[mask, 0], coords[mask, 1], s=size, color=color,
                           marker=marker, alpha=alpha, edgecolor="white", linewidth=0.35, zorder=3)
        mask = df["Group"].eq(group)
        pts = coords[mask]
        if len(pts) > 2:
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ell = Ellipse(
                pts.mean(axis=0),
                width=2.0 * np.sqrt(vals[0]),
                height=2.0 * np.sqrt(vals[1]),
                angle=angle,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=0.10,
                zorder=1,
            )
            ax.add_patch(ell)
            centroids[group] = pts.mean(axis=0)
            ax.scatter(pts.mean(axis=0)[0], pts.mean(axis=0)[1], s=64, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)
    if {"Saline", "METH"}.issubset(centroids):
        ax.annotate(
            "",
            xy=centroids["METH"],
            xytext=centroids["Saline"],
            arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": COLORS["ink"], "alpha": 0.55},
            zorder=2,
        )
    arrow_scale = 1.45
    label_offsets = {"VTA": (0.20, 0.22), "ACC": (0.31, -0.17), "NAco": (-0.16, -0.20)}
    for i, feature in enumerate(features):
        lx, ly = loadings[i] * arrow_scale
        ax.annotate(
            "",
            xy=(lx, ly),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": REGION_COLORS.get(feature, COLORS["ink"]),
                        "shrinkA": 0, "shrinkB": 0, "alpha": 0.86},
            zorder=5,
        )
        dx, dy = label_offsets.get(feature, (0.0, 0.0))
        ax.text(lx * 1.08 + dx, ly * 1.08 + dy, feature, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color=REGION_COLORS.get(feature, COLORS["ink"]),
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12, "alpha": 0.70})
    ax.axhline(0, color=COLORS["line"], linewidth=0.75)
    ax.axvline(0, color=COLORS["line"], linewidth=0.75)
    ax.set_xlabel("Discriminant embedding 1")
    ax.set_ylabel("Orthogonal embedding 2")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["gray"],
               markeredgecolor="white", label="Saline", markersize=5.2),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["teal"],
               markeredgecolor="white", label="METH", markersize=5.2),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["ink"],
               markeredgecolor="white", alpha=0.35, label="Synthetic", markersize=4.2),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.52, 1.01), ncol=3)
    ax.grid(color=COLORS["line"], linewidth=0.50, alpha=0.62)
    style_axes(ax)


def panel_b_faceted_sample_plane(fig: plt.Figure, spec) -> None:
    subgs = GridSpecFromSubplotSpec(2, 2, subplot_spec=spec, hspace=0.50, wspace=0.30)
    df = selected_features_file(1, 3).copy()
    df["GroupCode"] = df["Group"].map({"Saline": 0, "METH": 1})
    pairs = [("VTA", "ACC"), ("VTA", "NAco"), ("ACC", "NAco"), ("ACC", "VTA")]
    labels = ["VTA to ACC", "VTA to NAco", "ACC to NAco", "ACC to VTA"]
    for idx, ((xcol, ycol), lab) in enumerate(zip(pairs, labels)):
        ax = fig.add_subplot(subgs[idx // 2, idx % 2])
        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        xz = (x - np.mean(x)) / np.std(x, ddof=1)
        yz = (y - np.mean(y)) / np.std(y, ddof=1)
        for group, color, marker in [("Saline", COLORS["gray"], "o"), ("METH", COLORS["teal"], "o")]:
            s = df["Group"].eq(group)
            alpha = np.where(df.loc[s, "Origin"].eq("Synthetic"), 0.28, 0.90)
            sizes = np.where(df.loc[s, "Origin"].eq("Synthetic"), 12, 32)
            ax.scatter(xz[s], yz[s], s=sizes, color=color, alpha=alpha,
                       edgecolor="white", linewidth=0.35, marker=marker)
        for group, color in [("Saline", COLORS["gray"]), ("METH", COLORS["teal"])]:
            s = df["Group"].eq(group)
            cx = float(np.mean(xz[s]))
            cy = float(np.mean(yz[s]))
            cov = np.cov(xz[s], yz[s])
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ell = Ellipse((cx, cy), width=2.0 * np.sqrt(vals[0]), height=2.0 * np.sqrt(vals[1]),
                          angle=angle, facecolor=color, edgecolor=color, linewidth=1.0,
                          alpha=0.10 if group == "METH" else 0.08)
            ax.add_patch(ell)
            ax.scatter(cx, cy, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        ax.axhline(0, color=COLORS["line"], linewidth=0.70)
        ax.axvline(0, color=COLORS["line"], linewidth=0.70)
        ax.set_title(lab, pad=5.5, fontsize=7.6, fontweight="bold", color=COLORS["ink"])
        if idx // 2 == 1:
            ax.set_xlabel(f"{xcol} z-score")
        if idx % 2 == 0:
            ax.set_ylabel(f"{ycol} z-score")
        ax.set_xlim(-2.35, 3.55)
        ax.set_ylim(-2.25, 3.45)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(color=COLORS["line"], linewidth=0.45, alpha=0.60)
        style_axes(ax)

def estimation_plot(ax: plt.Axes, metric: str, ylabel: str, ylim: tuple[float, float], seed: int) -> None:
    df = perf_long()
    rng = np.random.default_rng(seed)
    x_positions = {"consensus": (0.0, 0.58), "fusion": (1.55, 2.13)}
    for scheme in ["consensus", "fusion"]:
        piv = df[df["scheme"].eq(scheme)].pivot(index="fold", columns="k", values=metric).sort_index()
        top1 = piv[1].to_numpy(dtype=float)
        top3 = piv[3].to_numpy(dtype=float)
        x1, x3 = x_positions[scheme]
        color = SCHEME_COLORS[scheme]
        for a, b in zip(top1, top3):
            ax.plot([x1, x3], [a, b], color=color, alpha=0.12, linewidth=0.85, zorder=1)
        ax.scatter(rng.normal(x1, 0.026, len(top1)), top1, s=17, color=color, alpha=0.52,
                   edgecolor="white", linewidth=0.30, zorder=3)
        ax.scatter(rng.normal(x3, 0.026, len(top3)), top3, s=20, color=color, alpha=0.72,
                   edgecolor="white", linewidth=0.30, zorder=3)
        for xx, vals in [(x1, top1), (x3, top3)]:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.add_patch(Rectangle((xx - 0.075, q1), 0.15, q3 - q1,
                                   facecolor="white", edgecolor=COLORS["axis"], linewidth=0.70, zorder=4))
            ax.plot([xx - 0.075, xx + 0.075], [med, med], color=COLORS["axis"], linewidth=0.95, zorder=5)
    ax.axvline(1.06, color=COLORS["line"], linewidth=0.9)
    ax.set_xticks([0.0, 0.58, 1.55, 2.13])
    ax.set_xticklabels(["Top-1", "Top-3", "Top-1", "Top-3"], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.28, 2.42)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.55, alpha=0.70)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["teal"],
               markeredgecolor="white", label="Consensus", markersize=5),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["coral"],
               markeredgecolor="white", label="Fused", markersize=5),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.52, 1.01))
    style_axes(ax)


def panel_c_metric_forest(ax: plt.Axes) -> None:
    df = perf_long()
    rand = pd.read_csv(ONEZONE / "random_roi_baseline.csv")
    rand_mean = float(rand[(rand["method"].eq("Ours")) & (rand["k"].eq(3))]["auc"].mean())
    rows = []
    for scheme in ["consensus", "fusion"]:
        for k in [1, 3, 5, 10]:
            rows.append((scheme, k))
    y = np.arange(len(rows))
    offsets = {"auc": -0.12, "auprc": 0.12}
    markers = {"auc": "o", "auprc": "s"}
    rng = np.random.default_rng(4215)
    for yi, (scheme, k) in enumerate(rows):
        color = SCHEME_COLORS[scheme]
        if k == 3:
            ax.axhspan(yi - 0.42, yi + 0.42, color=color, alpha=0.045, linewidth=0, zorder=0)
        for metric in ["auc", "auprc"]:
            vals = df[(df["scheme"].eq(scheme)) & (df["k"].eq(k))][metric].to_numpy()
            mean, lo, hi = bootstrap_mean_ci(vals, seed=4100 + yi + (0 if metric == "auc" else 40))
            yy = yi + offsets[metric]
            sample = vals if len(vals) <= 25 else rng.choice(vals, 25, replace=False)
            ax.scatter(sample, rng.normal(yy, 0.018, len(sample)), s=8,
                       color=color, alpha=0.18, edgecolor="none", zorder=2)
            ax.plot([lo, hi], [yy, yy], color=color, linewidth=1.15, alpha=0.70)
            ax.scatter(mean, yy, s=36, marker=markers[metric], color=color,
                       edgecolor="white", linewidth=0.65, zorder=3)
    ax.axvspan(0.985, 1.005, color=COLORS["teal"], alpha=0.035, linewidth=0, zorder=0)
    ax.axvline(rand_mean, color=COLORS["gray"], linestyle=(0, (3, 2)), linewidth=0.9, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'Cons.' if s == 'consensus' else 'Fused'} Top-{k}" for s, k in rows])
    ax.set_xlabel("Mean validation metric with bootstrap 95% CI")
    ax.set_xlim(0.775, 1.035)
    ax.set_ylim(len(rows) - 0.55, -0.55)
    handles = [
        Line2D([0], [0], marker="o", color=COLORS["ink"], markerfacecolor=COLORS["ink"],
               label="AUC", markersize=4.5, linewidth=0),
        Line2D([0], [0], marker="s", color=COLORS["ink"], markerfacecolor="white",
               label="AUPRC", markersize=4.5, linewidth=0),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.54, 1.01), ncol=2)
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.55, alpha=0.70)
    style_axes(ax)


def panel_d_auc_ecdf(ax: plt.Axes, show_legend: bool = True) -> None:
    df = perf_long()
    piv = {
        scheme: df[df["scheme"].eq(scheme)].pivot(index="fold", columns="k", values="auc").sort_index()
        for scheme in ["consensus", "fusion"]
    }
    order = np.argsort((piv["fusion"][3] - piv["fusion"][1]).to_numpy(dtype=float))[::-1]
    x = np.arange(1, len(order) + 1)
    consensus_top3 = piv["consensus"][3].to_numpy(dtype=float)[order]
    fused_top3 = piv["fusion"][3].to_numpy(dtype=float)[order]
    ax.axhspan(0.95, 1.025, color=COLORS["pale"], alpha=0.70, linewidth=0, zorder=0)
    ax.vlines(
        x,
        np.minimum(consensus_top3, fused_top3),
        np.maximum(consensus_top3, fused_top3),
        color=COLORS["gray"],
        alpha=0.24,
        linewidth=1.0,
        zorder=1,
    )
    for values, color, label in [
        (consensus_top3, COLORS["teal"], "Consensus Top-3"),
        (fused_top3, COLORS["coral"], "Fused Top-3"),
    ]:
        ax.plot(x, values, color=color, linewidth=1.75, alpha=0.96, label=label, zorder=3)
        ax.scatter(x, values, s=24, color=color, edgecolor="white", linewidth=0.45, zorder=4)
    ax.axhline(0.993, color=COLORS["gray"], linestyle=(0, (3, 2)), linewidth=0.8, alpha=0.70)
    ax.text(24.15, 1.014, "0.993", ha="left", va="bottom",
            fontsize=7.0, fontweight="bold", color=COLORS["gray"])
    style_profile_panel(
        ax,
        xlabel="Ranked cross-validation split",
        ylabel="Validation AUC",
        ylim=(0.45, 1.025),
        yticks=[0.50, 0.70, 0.90, 1.00],
    )
    handles = [
        Line2D([0], [0], color=COLORS["teal"], linewidth=1.8, label="Consensus Top-3"),
        Line2D([0], [0], color=COLORS["coral"], linewidth=1.8, label="Fused Top-3"),
    ]
    if show_legend:
        ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.01), ncol=2)


def panel_e_gain_waterfall(ax: plt.Axes, show_legend: bool = True) -> None:
    long = perf_long()
    piv = {
        scheme: long[long["scheme"].eq(scheme)].pivot(index="fold", columns="k", values="auc").sort_index()
        for scheme in ["consensus", "fusion"]
    }
    order = np.argsort(piv["fusion"][1].to_numpy(dtype=float))
    x = np.arange(1, len(order) + 1)
    ax.axhspan(0, 0.10, color=COLORS["pale"], alpha=0.70, linewidth=0, zorder=0)
    consensus_gain3 = (
        piv["consensus"][3].to_numpy(dtype=float)[order]
        - piv["consensus"][1].to_numpy(dtype=float)[order]
    )
    fusion_gain3 = (
        piv["fusion"][3].to_numpy(dtype=float)[order]
        - piv["fusion"][1].to_numpy(dtype=float)[order]
    )
    ax.vlines(
        x,
        np.minimum(consensus_gain3, fusion_gain3),
        np.maximum(consensus_gain3, fusion_gain3),
        color=COLORS["gray"],
        alpha=0.24,
        linewidth=1.0,
        zorder=1,
    )
    for values, color, label in [
        (consensus_gain3, COLORS["teal"], "Consensus Top-3 gain"),
        (fusion_gain3, COLORS["coral"], "Fused Top-3 gain"),
    ]:
        ax.plot(x, values, color=color, linewidth=1.75, label=label, zorder=3)
        ax.scatter(x, values, s=24, color=color, edgecolor="white", linewidth=0.45, zorder=4)
    ax.axhline(0, color=COLORS["axis"], linewidth=0.85)
    style_profile_panel(
        ax,
        xlabel="Ranked cross-validation split",
        ylabel="AUC gain vs Top-1",
        ylim=(-0.012, 0.525),
        yticks=[0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
    )
    if show_legend:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.54, 1.01),
                  ncol=2, handlelength=1.6, columnspacing=0.9)


def panel_f_random_percentile_cloud(ax: plt.Axes, show_legend: bool = True) -> None:
    actual = perf_long()
    actual = actual[(actual["scheme"].eq("fusion")) & (actual["k"].eq(3))].set_index("fold")["auc"]
    rand = pd.read_csv(ONEZONE / "random_roi_baseline.csv")
    rand = rand[(rand["method"].eq("Ours")) & (rand["k"].eq(3))]
    rows = []
    for fold, auc in actual.items():
        rr = rand[rand["fold"].eq(fold)]["auc"].to_numpy(dtype=float)
        strict = float(np.mean(rr < auc))
        ties = float(np.mean(rr == auc))
        rows.append((fold, strict, strict + 0.5 * ties, ties))
    tie_df = pd.DataFrame(rows, columns=["fold", "strict", "midrank", "ties"])
    tie_df = tie_df.sort_values(["midrank", "strict"]).reset_index(drop=True)
    x = np.arange(1, len(tie_df) + 1)
    strict = tie_df["strict"].to_numpy()
    midrank = tie_df["midrank"].to_numpy()
    ax.axhspan(0.50, 1.02, color=COLORS["pale"], alpha=0.70, linewidth=0, zorder=0)
    ax.vlines(x, strict, midrank, color=COLORS["gray"], alpha=0.24, linewidth=1.0, zorder=1)
    for values, color, label in [
        (strict, COLORS["coral"], "Strict"),
        (midrank, COLORS["plum"], "Tie-adjusted"),
    ]:
        ax.plot(x, values, color=color, linewidth=1.75, alpha=0.96, label=label, zorder=3)
        ax.scatter(x, values, s=24, color=color, edgecolor="white", linewidth=0.45, zorder=4)
    ax.axhline(0.50, color=COLORS["gray"], linestyle=(0, (3, 2)), linewidth=0.85)
    ax.text(24.58, 0.535, "50%", ha="left", va="center",
            fontsize=6.8, fontweight="bold", color=COLORS["gray"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.20, "alpha": 0.92})
    style_profile_panel(
        ax,
        xlabel="Ranked cross-validation split",
        ylabel="Percentile vs random Top-3",
        ylim=(-0.045, 1.02),
        yticks=[0.00, 0.25, 0.50, 0.75, 1.00],
    )
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    handles = [
        Line2D([0], [0], color=COLORS["coral"], linewidth=1.8, label="Strict"),
        Line2D([0], [0], color=COLORS["plum"], linewidth=1.8, label="Tie-adjusted"),
    ]
    if show_legend:
        ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.01),
                  ncol=2, columnspacing=0.90, handlelength=1.6)


def panel_g_top3_rank_dot(ax: plt.Axes) -> None:
    records = top3_records("fusion")
    ordered = sorted(records, key=lambda item: item[0])
    counts = Counter(region for _, regs in ordered for region in regs)
    main_regions = ["ACC", "VTA", "NAco", "BLA", "CLA", "DMS", "CeA"]
    x = np.arange(1, len(ordered) + 1)
    ax.axhspan(-0.18, 0.48, color=COLORS["pale"], alpha=0.80, linewidth=0, zorder=0)
    ax.axhspan(0.52, 3.18, color=COLORS["teal_light"], alpha=0.08, linewidth=0, zorder=0)
    for yline in [1, 2, 3]:
        ax.axhline(yline, color=COLORS["line"], linewidth=0.70, alpha=0.76, zorder=1)
    for region in main_regions:
        scores = []
        for _, regs in ordered:
            if region in regs:
                # Rank 1/2/3 are encoded as 3/2/1 so higher values mean higher Top-3 priority.
                scores.append(4 - (regs.index(region) + 1))
            else:
                scores.append(0)
        scores_arr = np.asarray(scores, dtype=float)
        color = REGION_COLORS.get(region, COLORS["gray"])
        is_major = counts.get(region, 0) >= 9
        ax.plot(
            x,
            scores_arr,
            color=color,
            linewidth=1.55 if is_major else 1.0,
            alpha=0.92 if is_major else 0.58,
            zorder=3 if is_major else 2,
        )
        ax.scatter(
            x,
            scores_arr,
            s=26 if is_major else 18,
            color=color,
            alpha=0.94 if is_major else 0.62,
            edgecolor="white",
            linewidth=0.45,
            zorder=4 if is_major else 3,
        )
    ax.set_xlim(1, 25)
    ax.set_ylim(-0.22, 3.22)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Not Top-3", "Rank 3", "Rank 2", "Rank 1"])
    ax.set_xlabel("Cross-validation split")
    ax.set_ylabel("Fused Top-3 profile")
    handles = [
        Line2D([0], [0], color=REGION_COLORS.get(region, COLORS["gray"]), marker="o",
               markeredgecolor="white", linewidth=1.45,
               markersize=4.4, label=f"{region} {counts.get(region, 0)}/25")
        for region in main_regions if counts.get(region, 0) > 0
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.50, 1.01),
              ncol=4, handlelength=1.0, handletextpad=0.35,
              columnspacing=0.70, borderaxespad=0.0, fontsize=7.0)
    ax.grid(axis="x", color=COLORS["line"], linewidth=0.42, alpha=0.42)
    style_axes(ax)


def panel_h_sorted_dominance(ax: plt.Axes) -> None:
    long = perf_long()
    actual = long[(long["scheme"].eq("fusion")) & (long["k"].eq(3))].sort_values("fold")
    rand = pd.read_csv(ONEZONE / "random_roi_baseline.csv")
    r = rand[(rand["method"].eq("Ours")) & (rand["k"].eq(3))]
    grouped = r.groupby("fold")["auc"]
    rmean = grouped.mean().reindex(actual["fold"]).to_numpy()
    r10 = grouped.quantile(0.10).reindex(actual["fold"]).to_numpy()
    r25 = grouped.quantile(0.25).reindex(actual["fold"]).to_numpy()
    r75 = grouped.quantile(0.75).reindex(actual["fold"]).to_numpy()
    r90 = grouped.quantile(0.90).reindex(actual["fold"]).to_numpy()
    a = actual["auc"].to_numpy()
    order = np.argsort(rmean)
    x = np.arange(1, len(order) + 1)
    ax.fill_between(x, r10[order], r90[order], color=COLORS["gray"], alpha=0.16, linewidth=0,
                    label="Random 10-90%")
    ax.fill_between(x, r25[order], r75[order], color=COLORS["gray"], alpha=0.28, linewidth=0,
                    label="Random 25-75%")
    ax.plot(x, rmean[order], color=COLORS["gray"], linewidth=1.35, label="Random mean")
    ax.fill_between(x, r90[order], a[order], where=a[order] >= r90[order],
                    color=COLORS["teal"], alpha=0.12, linewidth=0)
    ax.plot(x, a[order], color=COLORS["teal"], linewidth=2.0, label="Fused Top-3")
    ax.scatter(x, a[order], s=24, color=COLORS["teal"], edgecolor="white", linewidth=0.45, zorder=4)
    ax.scatter(x[a[order] < r90[order]], a[order][a[order] < r90[order]], s=42,
               color=COLORS["coral"], edgecolor="white", linewidth=0.55, zorder=5)
    ax.set_xlabel("Splits ordered by random-baseline AUC")
    ax.set_ylabel("Validation AUC")
    ax.set_xlim(1, 25)
    ax.set_ylim(0.62, 1.025)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.grid(axis="y", color=COLORS["line"], linewidth=0.55, alpha=0.70)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.54, 1.01), ncol=4,
              handlelength=1.5, columnspacing=0.9)
    style_axes(ax)


def make_figures() -> None:
    single_specs = [
        ("premium_panel_A_lowdim_embedding", panel_a_lowdim_embedding, (6.5, 3.65)),
        ("premium_panel_C_metric_forest", panel_c_metric_forest, (5.0, 5.0)),
        ("premium_panel_D_auc_ecdf", panel_d_auc_ecdf, (5.2, 3.4)),
        ("premium_panel_E_gain_waterfall", panel_e_gain_waterfall, (5.2, 3.4)),
        ("premium_panel_G_top3_rank_dot", panel_g_top3_rank_dot, (6.3, 2.9)),
        ("premium_panel_H_sorted_dominance", panel_h_sorted_dominance, (7.0, 3.2)),
    ]
    for stem, func, size in single_specs:
        fig, ax = plt.subplots(figsize=size)
        func(ax)
        if stem == "premium_panel_C_metric_forest":
            fig.subplots_adjust(left=0.30, right=0.97, bottom=0.15, top=0.86)
            save_all(fig, stem, bbox_inches=None)
        else:
            save_all(fig, stem)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    panel_f_random_percentile_cloud(ax)
    save_all(fig, "premium_panel_F_random_percentile_cloud")

    fig = plt.figure(figsize=(5.0, 5.0))
    panel_b_faceted_sample_plane(fig, fig.add_gridspec(1, 1)[0])
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.94)
    save_all(fig, "premium_panel_B_faceted_sample_plane", bbox_inches=None)

    screenshot_order_specs = [
        ("panel_F_gain_profile", panel_e_gain_waterfall, (5.2, 3.4)),
        ("panel_G_auc_profile", panel_d_auc_ecdf, (5.2, 3.4)),
        ("panel_H_random_percentile_profile", panel_f_random_percentile_cloud, (5.2, 3.4)),
    ]
    for stem, func, size in screenshot_order_specs:
        fig, ax = plt.subplots(figsize=size)
        func(ax)
        save_all(fig, stem)

    fig = plt.figure(figsize=(15.6, 13.4))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.72,
        wspace=0.48,
        height_ratios=[1.16, 1.06, 1.06, 0.92],
    )
    panel_a_lowdim_embedding(fig.add_subplot(gs[0, 0:2]))
    panel_c_metric_forest(fig.add_subplot(gs[0, 2]))
    panel_b_faceted_sample_plane(fig, gs[1:3, 0])
    panel_g_top3_rank_dot(fig.add_subplot(gs[1, 1:3]))
    panel_h_sorted_dominance(fig.add_subplot(gs[2, 1:3]))
    panel_e_gain_waterfall(fig.add_subplot(gs[3, 0]))
    panel_d_auc_ecdf(fig.add_subplot(gs[3, 1]))
    panel_f_random_percentile_cloud(fig.add_subplot(gs[3, 2]))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"figure_3_5_discriminatory_info_v3_combined.{ext}", dpi=600, bbox_inches="tight")
    fig.savefig(
        OUT / "figure_3_5_discriminatory_info_v3_combined_300dpi.tif",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    perf_summary().to_csv(OUT / "figure_3_5_performance_summary.csv", index=False)
    zip_path = ROOT / "figure_3_5_discriminatory_info_v3_300dpi_tiff.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for path in sorted(OUT.glob("*")):
            if path.suffix.lower() in {".tif", ".tiff", ".png", ".pdf", ".svg", ".csv"}:
                zf.write(path, arcname=f"{OUT.name}/{path.name}")


def main() -> None:
    set_style()
    make_figures()
    print(f"Saved Figure 3.5 v3 panels to {OUT.resolve()}")


if __name__ == "__main__":
    main()
