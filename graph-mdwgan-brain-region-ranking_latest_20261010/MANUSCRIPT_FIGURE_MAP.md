# Manuscript figure map

The historical analysis filenames use section numbers (`3.1`-`3.7`). The current manuscript labels are offset by one.

| Manuscript item | Plot script | Output directory |
|---|---|---|
| Figure 1 data panels | `scripts/figures/plot_figure_1_heatmap.py` and `plot_figure_1_preprocessing.py` | `results/figures/Figure_1_assets/` |
| Figure 2 | `scripts/figures/plot_figure_3_1_premium.py` | `results/figures/figure_3_1_premium/` |
| Figure 3 | `scripts/figures/plot_figure_3_2_advanced.py` | `results/figures/figure_3_2_advanced/` |
| Figure 4 | `scripts/figures/plot_figure_3_3_second_tier_bio_upgrade.py` | `results/figures/figure_3_3_second_tier_bio_upgrade_300dpi/` |
| Figure 5 | `scripts/figures/plot_figure_3_4_advanced_common.py` | `results/figures/figure_3_4_advanced_common_300dpi/` |
| Figure 6 | `scripts/figures/plot_figure_3_5_discriminatory_info_v3.py` | `results/figures/figure_3_5_discriminatory_info_v3/` |
| Figure 7 | `scripts/figures/plot_figure_3_6_robustness.py` | `results/figures/figure_3_6_robustness/` |
| Figure 8 | `scripts/figures/plot_figure_3_7_literature_concordance.py` | `results/figures/figure_3_7_literature_concordance/` |
| Supplementary Figure S3 | `scripts/figures/plot_figure_3_6_supplement_4panel_premium.py` | `results/figures/figure_3_6_supplement_4panel_premium/` |
| Supplementary Figure S4 | `scripts/figures/plot_figure_3_7_supplement_common.py` and `plot_figure_3_7_supplement_DEH.py` | `results/figures/figure_3_7_supplement_*/` |

Run all code-generated figures with:

```bash
python scripts/figures/run_all_figures.py
```
