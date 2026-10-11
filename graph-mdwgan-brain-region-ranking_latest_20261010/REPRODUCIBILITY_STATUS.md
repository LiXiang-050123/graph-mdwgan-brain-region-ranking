# Reproducibility status (2026-10-10)

## Correction incorporated

The secondary resampling scripts now calculate consensus ranks from exactly five feature-selection methods: ANOVA, mutual information, L1-logistic regression, random forest and RFE-SVM. GNN-lite is excluded from consensus and incorporated once, with the specified weight, to form the fused ranking.

The shared implementation is `scripts/ranking_utils.py`. `tests/test_consensus_gnn_weighting.py` rejects a `base_scores` mapping that contains a GNN entry.

## Frozen corrected results

- `results/reference/corrected_secondary/`: corrected performance, stability and frequency tables.
- `results/reference/cv_results/`: compact fold-level tables needed for Figures 3-6; checkpoints and redundant images are intentionally omitted.
- `results/reference/seed_quality/`: CSV/JSON quality outputs and representative synthetic data; model checkpoints are omitted.
- `results/reference/pubmed_concordance_corrected/`: frozen PubMed snapshot and corrected concordance outputs.
- `results/reference/low_data/`: classifier-stage downsampling outputs.

## Verified values

- Corrected fused Top-3 Jaccard: 2x `0.393`, 4x `0.438`, 8x `0.464`.
- Corrected literature concordance: Spearman rho `0.525827`, parametric P `0.025006`, permutation P `0.023298` (10,000 permutations; seed `20250829`).
- Standardized corrected Top-3 AUC: Graph-MDWGAN `0.970 +/- 0.086`; RealOnly `0.977 +/- 0.082`; ROS `0.917 +/- 0.140`; SMOTE `0.943 +/- 0.110`.

These standardized estimates are not directly comparable with the primary nested Top-k evaluation.

Fresh outputs from the repository verification run are stored in `results/verification/`.

## Interpretation limits

The low-data analysis downsamples real observations at the classifier stage while using a generator trained on the complete outer-training fold. It is not a strict retraining test of generator performance at each retained-data fraction. Repeated-split intervals are descriptive because animals recur across splits. Synthetic samples are used for regularization and ranking sensitivity and are not treated as new biological observations.
