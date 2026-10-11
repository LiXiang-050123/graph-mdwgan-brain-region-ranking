from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "data" / "brain_region_cfos_with_paradigm.csv"
OUT = REPO_ROOT / "results" / "figures" / "Figure_1_assets"
OUT.mkdir(parents=True, exist_ok=True)
REGIONS = [
    "ACC", "AM", "BLA", "CeA", "CLA", "CM", "DMS", "GP", "IL",
    "LH", "MS", "NAco", "NAsh", "OFC", "PL", "RSC", "VP", "VTA",
]


def load_matrix() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(SOURCE)
    data = data.set_index("Sample ID")
    matrix = data[REGIONS].apply(pd.to_numeric, errors="raise")
    groups = data["Group"].astype(str)
    if matrix.shape != (24, 18):
        raise RuntimeError(f"Expected 24 samples x 18 regions, found {matrix.shape}")
    return matrix, groups


def draw(matrix: pd.DataFrame, groups: pd.Series) -> None:
    # Standardize each region across animals so regional baselines do not dominate.
    z = matrix.sub(matrix.mean(axis=0), axis=1).div(matrix.std(axis=0, ddof=1), axis=1)
    plot_data = z.T

    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    cmap = LinearSegmentedColormap.from_list(
        "cfos_blue",
        ["#F7FBFF", "#D7E9F5", "#9CC9E2", "#4A96C5", "#1764A0", "#083B73"],
        N=256,
    )
    group_colors = {"Saline": "#94BEDB", "METH": "#1E5C91"}

    fig = plt.figure(figsize=(7.2, 4.8), facecolor="none")
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1, 0.045],
        height_ratios=[0.17, 0.11, 1],
        hspace=0.035,
        wspace=0.09,
        left=0.105,
        right=0.91,
        bottom=0.22,
        top=0.92,
    )

    title_ax = fig.add_subplot(grid[0, 0])
    title_ax.set_facecolor("none")
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.62,
        "24 samples × 18 brain regions",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#15191E",
    )

    annotation_ax = fig.add_subplot(grid[1, 0])
    annotation_ax.set_facecolor("none")
    for index, group in enumerate(groups):
        annotation_ax.add_patch(
            plt.Rectangle((index - 0.5, -0.5), 1, 1, color=group_colors[group], linewidth=0)
        )
    annotation_ax.axvline(10.5, color="white", linewidth=2)
    annotation_ax.set_xlim(-0.5, len(groups) - 0.5)
    annotation_ax.set_ylim(-0.5, 0.5)
    annotation_ax.axis("off")
    annotation_ax.text(5, 0, "Saline (n=11)", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    annotation_ax.text(17, 0, "METH (n=13)", ha="center", va="center", color="white", fontsize=7, fontweight="bold")

    heat_ax = fig.add_subplot(grid[2, 0])
    heat_ax.set_facecolor("none")
    image = heat_ax.imshow(plot_data.to_numpy(), cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
    heat_ax.set_xticks(np.arange(len(plot_data.columns)))
    heat_ax.set_xticklabels(plot_data.columns, rotation=90, ha="center", va="top", fontsize=5.8)
    heat_ax.set_yticks(np.arange(len(plot_data.index)))
    heat_ax.set_yticklabels(plot_data.index, fontsize=7.5)
    heat_ax.set_xlabel("Animals", fontsize=8.5, fontweight="bold", labelpad=6)
    heat_ax.set_ylabel("Brain regions", fontsize=8.5, fontweight="bold", labelpad=6)
    heat_ax.tick_params(axis="both", length=0, colors="#20252B")
    heat_ax.axvline(10.5, color="white", linewidth=2)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    color_ax = fig.add_subplot(grid[2, 1])
    color_ax.set_facecolor("none")
    colorbar = fig.colorbar(image, cax=color_ax)
    colorbar.set_ticks([-2, -1, 0, 1, 2])
    colorbar.ax.tick_params(labelsize=7, length=2, colors="#20252B")
    colorbar.set_label("Relative c-Fos density\n(row z-score)", fontsize=7.5, labelpad=6)
    colorbar.outline.set_linewidth(0.6)
    colorbar.outline.set_edgecolor("#56606A")

    raw_export = matrix.copy()
    raw_export.insert(0, "Group", groups)
    raw_export.to_csv(OUT / "cfos_24x18_raw_matrix.csv", encoding="utf-8-sig")
    z_export = z.copy()
    z_export.insert(0, "Group", groups)
    z_export.to_csv(OUT / "cfos_24x18_row_zscores.csv", encoding="utf-8-sig")

    metadata = {
        "source_file": str(SOURCE),
        "source_sheet": "Fig1d and Fig1e reconstruction",
        "n_samples": matrix.shape[0],
        "n_regions": matrix.shape[1],
        "saline_n": int((groups == "Saline").sum()),
        "meth_n": int((groups == "METH").sum()),
        "standardization": "per-region z-score across all 24 animals; sample SD (ddof=1)",
    }
    pd.Series(metadata, name="value").to_csv(OUT / "cfos_24x18_heatmap_metadata.tsv", sep="\t", header=True)

    for suffix in ("png", "tif", "pdf"):
        tiff_options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tif" else {}
        fig.savefig(
            OUT / f"cfos_24x18_heatmap_ZR_blue_transparent.{suffix}",
            dpi=600 if suffix != "pdf" else None,
            transparent=True,
            facecolor="none",
            edgecolor="none",
            **tiff_options,
        )
    plt.close(fig)


if __name__ == "__main__":
    draw(*load_matrix())
