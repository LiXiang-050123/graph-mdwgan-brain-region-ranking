# Code map

| Manuscript component | Script |
|---|---|
| Graph-MDWGAN architecture, losses, training and checkpointing | `scripts/graph_mdwgan.py` |
| Single-split augmentation and five-seed quality assessment | `scripts/run_single_split_augmentation.py`, `scripts/run_quality_5seeds.py` |
| Five feature-selection methods and GNN-lite integration | `scripts/feature_selection.py` |
| Repeated 5-fold cross-validation and primary Top-k evaluation | `scripts/run_primary_cv.py` |
| Extended bootstrap stability analysis and ranking used for Figure 8 | `scripts/run_extended_resampling.py` |
| ROS/SMOTE/RealOnly baseline and classifier/factor sensitivity | `scripts/run_baseline_sensitivity.py` |
| Low-data stress test using fold-specific synthetic pools | `scripts/run_low_data_stress.py` |
| PubMed literature-evidence concordance | `scripts/run_pubmed_concordance.py` |
| Offline bibliometric summaries | `scripts/run_bibliometric_enrichment.py` |

The repository contains only the final analysis path. Historical experiments, duplicate plotting scripts, cached checkpoints, virtual environments and outdated Figure 8 code were omitted.
