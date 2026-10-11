# Data

`source/Fig1.xlsx` contains the source-publication sheets. `scripts/reconstruct_analysis_matrix.py` reads `Fig1d` and `Fig1e`, writes the labeled animal-by-region table `brain_region_cfos_with_paradigm.csv`, and verifies that its pooled values exactly reproduce `brain_region_cfos.xlsx`.

`brain_region_cfos.xlsx` is retained in the original region-by-animal layout expected by the analysis scripts. `brain_region_cfos_reconstructed.xlsx` is generated as an independent reconstruction check.

The matrix contains 11 saline-treated and 13 methamphetamine-treated rats across two behavioral paradigms. Self-administration c-Fos quantification contributes 6 saline and 7 METH animals; cue-induced reinstatement contributes 5 saline and 6 METH animals. The pooled treatment analysis is exploratory, and the paradigm labels should be retained for stratified checks. See the manuscript for the source-study citation and reconstruction details.
