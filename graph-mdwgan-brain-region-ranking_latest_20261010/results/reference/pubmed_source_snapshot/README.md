# PubMed Literature Concordance Output

This folder contains a reproducible PubMed-based literature evidence concordance
analysis for methamphetamine-related candidate brain regions.

Files:

- `pubmed_literature_concordance.py`: analysis script in the parent folder.
- `pubmed_queries.json`: PubMed search date, search strategy, per-region queries, and hit counts.
- `pubmed_unique_records.csv`: deduplicated PubMed records retrieved across all candidate brain regions.
- `pubmed_region_records_long.csv`: long-format region-record table; the same PMID can appear under multiple brain regions.
- `literature_evidence_summary.csv`: brain-region-level evidence metrics and concordance classes.
- `ml_ranking_all_candidate_regions.csv`: machine-learning ranking imported from the project result file.
- `concordance_metrics.json`: Spearman, Kendall, Top-k overlap, and permutation-test results.
- `figure_concordance_scatter.png`: scatter/bubble figure for manuscript use.
- `PubMed文献证据一致性分析报告.md`: Chinese summary report with method text draft.
- `文献计量学增强分析报告.md`: expanded Chinese bibliometric report with manuscript-ready method/results text.
- `文献计量学框架优化说明.md`: lessons borrowed from related bibliometric/scientometric studies and how they were adapted here.
- `bibliometric_annual_publication_trend.csv` and `figure_annual_publication_trend.png`: annual publication trend.
- `bibliometric_top_journals.csv` and `figure_top_journals.png`: journal distribution.
- `bibliometric_top_mesh_terms.csv` and `figure_top_mesh_terms.png`: MeSH term frequency.
- `bibliometric_evidence_type_by_region.csv` and `figure_evidence_type_by_region.png`: automated evidence-type coding by brain region.
- `bibliometric_species_distribution.csv` and `figure_species_distribution.png`: automated species coding.
- `bibliometric_region_cooccurrence_edges.csv`: brain-region co-occurrence edge list for network visualization.
- `bibliometric_mesh_cooccurrence_edges.csv`: MeSH co-occurrence edge list for VOSviewer/Gephi/Cytoscape-style visualization.

Interpretation note:

The analysis evaluates concordance between machine-learning prioritization and
existing literature evidence. It should not be described as proof of model
accuracy or proof of causal involvement of a brain region.

Recommended manuscript framing:

Use the enhanced bibliometric outputs to describe the scope and structure of the
methamphetamine brain-region literature, then use the concordance metrics to
evaluate whether machine-learning-prioritized regions are consistent with or
complementary to prior research.
