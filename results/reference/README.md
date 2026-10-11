# Reference results

The current manuscript snapshot is organized by analysis stage:

- `corrected_secondary/`: corrected five-method consensus and single-use GNN fusion results.
- `corrected_secondary_ours/` and `corrected_secondary_baselines/`: source batches used to create the merged corrected tables.
- `cv_results/`: compact fold-level primary tables required by plotting code.
- `seed_quality/`: five-seed quality tables and representative generated samples.
- `low_data/`: classifier-stage downsampling results.
- `pubmed_concordance_corrected/`: frozen literature snapshot and corrected concordance.
- `pubmed_source_snapshot/`: pre-ranking frozen literature tables used for deterministic recalculation.
- `robustness_sources/` and `ros_fixunique/`: source tables for Figure 7 and Supplementary Figure S3.

Corrected literature statistics are Spearman rho `0.5258267269`, parametric P `0.0250060675` and permutation P `0.0232976702` (10,000 permutations; seed `20250829`).

`legacy_original_release/` contains files copied from the earlier public package. They are retained only for provenance and must not be used as current manuscript values.
