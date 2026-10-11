from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "data" / "brain_region_cfos_with_paradigm.csv"
OUT = REPO_ROOT / "results" / "figures" / "Figure_1_assets"
OUT.mkdir(parents=True, exist_ok=True)
REGION = "ACC"


def load_acc() -> pd.DataFrame:
    data = pd.read_csv(SOURCE, usecols=["Sample ID", "Group", REGION])
    data[REGION] = pd.to_numeric(data[REGION], errors="raise")
    data["log1p_ACC"] = np.log1p(data[REGION])
    if len(data) != 24:
        raise RuntimeError(f"Expected 24 animals, found {len(data)}")
    return data


def draw(data: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })
    dark_blue = "#1E5C91"
    mid_blue = "#4A96C5"
    light_blue = "#94BEDB"
    text = "#171B20"
    group_colors = {"Saline": light_blue, "METH": dark_blue}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), facecolor="none")
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.23, top=0.78, wspace=0.38)
    fig.suptitle(
        "ACC c-Fos distribution before and after transformation",
        x=0.5,
        y=0.94,
        fontsize=10,
        fontweight="bold",
        color=text,
    )

    panels = [
        ("ACC", "Raw", "c-Fos density"),
        ("log1p_ACC", "log(1+x)", "log(1 + c-Fos density)"),
    ]
    rng = np.random.default_rng(20260826)
    for ax, (column, title, xlabel) in zip(axes, panels):
        values = data[column].to_numpy(float)
        counts, bins, patches = ax.hist(
            values,
            bins=7,
            color=mid_blue,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.96,
        )
        ymax = max(counts.max(), 1)
        jitter = rng.uniform(-0.035, 0.035, size=len(values)) * ymax
        for group in ["Saline", "METH"]:
            mask = data["Group"].eq(group).to_numpy()
            ax.scatter(
                values[mask],
                np.full(mask.sum(), -0.11 * ymax) + jitter[mask],
                s=20,
                c=group_colors[group],
                edgecolors="white",
                linewidths=0.45,
                zorder=5,
                clip_on=False,
            )
        ax.axvline(np.median(values), color="#173F63", linewidth=1.2, linestyle="--")
        ax.text(
            np.median(values),
            ymax * 1.015,
            "median",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#173F63",
        )
        ax.set_title(title, fontsize=9, fontweight="bold", color=text, pad=5)
        ax.set_xlabel(xlabel, fontsize=8, fontweight="bold", color=text, labelpad=5)
        ax.set_ylabel("Count", fontsize=8, fontweight="bold", color=text)
        ax.tick_params(axis="both", labelsize=7, length=3, width=0.7, colors=text)
        ax.set_ylim(-0.18 * ymax, ymax * 1.18)
        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#49545E")
        ax.spines["bottom"].set_color("#49545E")

    axes[0].annotate(
        "",
        xy=(1.13, 0.55),
        xytext=(1.01, 0.55),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=dark_blue, lw=1.5),
        annotation_clip=False,
    )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=light_blue,
                   markeredgecolor="white", label="Saline (n=11)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=dark_blue,
                   markeredgecolor="white", label="METH (n=13)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.855),
        ncol=2,
        frameon=False,
        fontsize=7,
        handletextpad=0.4,
        columnspacing=1.4,
    )

    export = data.rename(columns={"ACC": "ACC_raw", "log1p_ACC": "ACC_log1p"})
    export.to_csv(OUT / "ACC_raw_vs_log1p_values.csv", index=False, encoding="utf-8-sig")
    metadata = pd.Series({
        "source_file": str(SOURCE),
        "source_sheet": "Fig1d and Fig1e reconstruction",
        "brain_region": REGION,
        "n_samples": len(data),
        "saline_n": int(data["Group"].eq("Saline").sum()),
        "meth_n": int(data["Group"].eq("METH").sum()),
        "transformation": "natural logarithm log(1+x), numpy.log1p",
        "raw_median": float(data[REGION].median()),
        "log1p_median": float(data["log1p_ACC"].median()),
    }, name="value")
    metadata.to_csv(OUT / "ACC_raw_vs_log1p_metadata.tsv", sep="\t", header=True)

    for suffix in ("png", "tif", "pdf"):
        tiff_options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tif" else {}
        fig.savefig(
            OUT / f"ACC_raw_vs_log1p_ZR_blue_transparent.{suffix}",
            dpi=600 if suffix != "pdf" else None,
            transparent=True,
            facecolor="none",
            edgecolor="none",
            **tiff_options,
        )
    plt.close(fig)


if __name__ == "__main__":
    draw(load_acc())
