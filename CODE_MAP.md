# Code map

| Analysis component | Script |
|---|---|
| Graph-MDWGAN architecture, training and checkpointing | `scripts/graph_mdwgan.py` |
| Reconstruction of Fig. 1d/1e with paradigm labels | `scripts/reconstruct_analysis_matrix.py` |
| Single-split augmentation | `scripts/run_single_split_augmentation.py` |
| Five-seed quality assessment | `scripts/run_quality_5seeds.py` |
| Five-method feature-selection consensus and primary GNN integration | `scripts/feature_selection.py` |
| Repeated outer cross-validation and primary Top-k evaluation | `scripts/run_primary_cv.py` |
| Shared corrected consensus/fusion implementation | `scripts/ranking_utils.py` |
| Corrected extended bootstrap stability analysis | `scripts/run_extended_resampling.py` |
| Corrected ROS/SMOTE/RealOnly and factor/classifier sensitivity | `scripts/run_baseline_sensitivity.py` |
| Classifier-stage low-data downsampling | `scripts/run_low_data_stress.py` |
| Frozen-snapshot corrected concordance recalculation | `scripts/recompute_corrected_concordance.py` |
| Live PubMed concordance and bibliometrics | `scripts/run_pubmed_concordance.py`, `scripts/run_bibliometric_enrichment.py` |
| Merge split corrected result batches | `scripts/merge_corrected_secondary_results.py` |
| All manuscript plot regeneration | `scripts/figures/run_all_figures.py` |

The regression test in `tests/test_consensus_gnn_weighting.py` verifies that changing GNN-lite does not change the five-method consensus, does change fusion and is rejected if inserted into the base-method mapping.

The source filenames for plotting retain the historical section numbering. Use `MANUSCRIPT_FIGURE_MAP.md` for current manuscript Figure 2-8 labels.
