"""Regenerate manuscript Figures 2-8 and the corrected supplementary panels."""

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "plot_figure_1_heatmap.py",
    "plot_figure_1_preprocessing.py",
    "plot_figure_3_1_premium.py",  # manuscript Figure 2
    "plot_figure_3_2_advanced.py",  # manuscript Figure 3
    "plot_figure_3_3_second_tier_bio_upgrade.py",  # manuscript Figure 4
    "plot_figure_3_4_advanced_common.py",  # manuscript Figure 5
    "plot_figure_3_5_discriminatory_info_v3.py",  # manuscript Figure 6
    "plot_figure_3_6_robustness.py",  # manuscript Figure 7
    "plot_figure_3_7_literature_concordance.py",  # manuscript Figure 8
    "plot_figure_3_6_supplement_4panel_premium.py",
    "plot_figure_3_7_supplement_common.py",
    "plot_figure_3_7_supplement_DEH.py",
)


def main() -> None:
    for script in SCRIPTS:
        print(f"[figures] running {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], check=True)


if __name__ == "__main__":
    main()
