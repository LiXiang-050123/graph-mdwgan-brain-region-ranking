# Graph-MDWGAN brain-region ranking

Core code, corrected result snapshots and manuscript figure scripts for the study of machine-learning prioritization of methamphetamine-associated brain regions.

This 2026-10-10 release incorporates the GNN-lite weighting correction. Consensus is calculated from five methods (ANOVA, mutual information, L1-logistic regression, random forest and RFE-SVM); GNN-lite is then added exactly once to form the fused ranking.

## Repository layout

```text
data/                         24-animal x 18-region analysis input
scripts/                      model training and statistical workflows
scripts/figures/              current Figure 2-8 and supplementary plot code
tests/                        correction-focused regression test
results/reference/            compact frozen corrected tables
results/figures/              publication outputs and regenerated figures
outputs/                      destination for a fresh full rerun
```

See `CODE_MAP.md`, `MANUSCRIPT_FIGURE_MAP.md` and `REPRODUCIBILITY_STATUS.md` for the exact script-to-result mapping and corrected reference values.

## Installation

Python 3.11 is recommended. GPU acceleration is optional but substantially reduces Graph-MDWGAN training time.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate meth-brain-region-prioritization
```

## Quick verification

Run from the repository root:

```bash
python scripts/reconstruct_analysis_matrix.py
python tests/test_consensus_gnn_weighting.py
python scripts/figures/run_all_figures.py
```

The plotting command reads only repository-relative files under `results/reference/` and writes regenerated output under `results/figures/`. It does not depend on the original desktop folder names.

Directories named `Figure_2` through `Figure_8` contain the curated latest exports copied from the manuscript workspace. Lowercase `figure_3_1_*` through `figure_3_7_*` directories are fresh outputs produced during repository verification; the manuscript mapping is documented in `MANUSCRIPT_FIGURE_MAP.md`.

## Full analysis

### 1. Synthetic-data quality

```bash
python scripts/run_quality_5seeds.py
```

### 2. Primary repeated cross-validation

```bash
python scripts/run_primary_cv.py --strict_ours --keep_fold_artifacts
```

Defaults reproduce repeated stratified 5-fold cross-validation repeated five times, factor-8 augmentation, a 500-tree random forest, GNN-lite neighborhood size 5 and a GNN fusion weight of 1/6. Augmentation, graph construction, feature ranking and scaling are restricted to each outer-training fold.

### 3. Corrected extended resampling

```bash
python scripts/run_extended_resampling.py \
  --excel data/brain_region_cfos.xlsx \
  --reuse_cv_dir outputs/cv_results \
  --out_dir outputs/extended_resampling \
  --cache_bootstrap
```

### 4. Corrected standardized baseline analysis

```bash
python scripts/run_baseline_sensitivity.py \
  --excel data/brain_region_cfos.xlsx \
  --reuse_cv_dir outputs/cv_results \
  --out_dir outputs/baseline_sensitivity
```

This protocol is separate from the primary nested Top-k evaluation; its AUC values should not be compared directly with the primary estimates.

### 5. Low-data sensitivity analysis

```bash
python scripts/run_low_data_stress.py \
  --base_dir . \
  --reuse_cv_dir outputs/cv_results \
  --excel data/brain_region_cfos.xlsx \
  --do_quick 1
```

This is classifier-stage real-sample downsampling with a generator pretrained on the complete outer-training fold. It is not evidence that the generator was trained using only 20-80% of the real observations.

### 6. Literature concordance

For a deterministic check against the frozen literature snapshot:

```bash
python scripts/recompute_corrected_concordance.py
```

For a new PubMed search:

```bash
python scripts/run_pubmed_concordance.py \
  --ml_ranking outputs/extended_resampling/freq_top10_factor8_Ours_consensus_raw.csv \
  --out_dir outputs/pubmed_concordance
python scripts/run_bibliometric_enrichment.py --out_dir outputs/pubmed_concordance
```

PubMed changes over time, so a live search need not reproduce the frozen record counts.

## Data and interpretation

The 24-animal matrix combines condition-specific source-data sheets from self-administration and cue-induced reinstatement, with paradigm labels retained in the reconstruction. The pooled treatment analysis is exploratory; paradigm-stratified analyses are needed when interpreting whether priorities generalize across behavioral contexts.

Repeated-split intervals summarize empirical variation across 25 splits. Because the same animals occur in multiple splits, they are descriptive rather than independent-sample confidence intervals. Synthetic observations are used as a regularization and ranking-sensitivity device, not as new biological measurements.

## Provenance

The corrected result manifest is `results/reference/corrected_secondary/CORRECTED_RUN_MANIFEST.json`. `SHA256SUMS.txt` records file checksums for this release. Historical experiments, virtual environments and large checkpoint caches are intentionally excluded; a full rerun recreates checkpoints under `outputs/`.
