# Machine learning-based prioritization of methamphetamine-associated brain regions

This repository contains the core analysis code for the manuscript **“Machine learning-based prioritization of candidate brain regions associated with methamphetamine addiction.”**

The workflow analyzes c-Fos measurements from 24 rats across 18 brain regions. Graph-MDWGAN augmentation is restricted to each outer training fold, five feature-selection methods are combined into a consensus ranking, and a GNN-lite structural score is integrated using fixed weights. Repeated cross-validation, bootstrap stability analysis, conventional oversampling baselines, low-data stress testing and PubMed concordance are implemented as separate steps.

## Repository structure

```text
.
├── data/
│   └── brain_region_cfos.xlsx
├── scripts/
│   ├── graph_mdwgan.py
│   ├── feature_selection.py
│   ├── run_single_split_augmentation.py
│   ├── run_quality_5seeds.py
│   ├── run_primary_cv.py
│   ├── run_extended_resampling.py
│   ├── run_baseline_sensitivity.py
│   ├── run_low_data_stress.py
│   ├── run_pubmed_concordance.py
│   └── run_bibliometric_enrichment.py
├── results/reference/
├── outputs/
├── CODE_MAP.md
├── requirements.txt
└── environment.yml
```

## Installation

Python 3.10–3.12 is recommended. GPU acceleration is optional but substantially reduces Graph-MDWGAN training time.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

A Conda environment is also provided:

```bash
conda env create -f environment.yml
conda activate meth-brain-region-prioritization
```

## Reproducing the analysis

Run all commands from the repository root.

### 1. Five-seed synthetic-data quality analysis

```bash
python scripts/run_quality_5seeds.py
```

This step evaluates the factor-8 Graph-MDWGAN outputs across seeds `20250829`–`20250833` and compares their distributional profiles with ROS and SMOTE. The Graph-MDWGAN distance metrics use the model-specific calibrated outputs; duplicate checks use direct generator outputs.

### 2. Primary repeated cross-validation

```bash
python scripts/run_primary_cv.py \
  --strict_ours \
  --keep_fold_artifacts
```

Default settings match the manuscript: repeated stratified 5-fold cross-validation repeated five times, factor 8, random-forest ranking with 500 trees, GNN-lite neighborhood size 5, and consensus/GNN weights of 5/6 and 1/6.

The command writes fold-specific checkpoints and rankings to `outputs/cv_results`. These artifacts are reused by later sensitivity analyses.

### 3. Extended resampling and stability ranking

```bash
python scripts/run_extended_resampling.py \
  --reuse_cv_dir outputs/cv_results \
  --cache_bootstrap
```

This produces the cumulative bootstrap Top-10 frequencies used for the machine-learning ordering in the Figure 8 literature-concordance analysis.

### 4. Standardized baseline and sensitivity analysis

```bash
python scripts/run_baseline_sensitivity.py \
  --reuse_cv_dir outputs/cv_results
```

The analysis compares Graph-MDWGAN, ROS, SMOTE and RealOnly across augmentation factors, duplicate-handling conditions and three classifiers. It is evaluated separately from the primary Top-k analysis.

### 5. Low-data stress test

```bash
python scripts/run_low_data_stress.py \
  --base_dir . \
  --reuse_cv_dir outputs/cv_results \
  --excel data/brain_region_cfos.xlsx \
  --do_quick 1
```

The default quick mode reproduces the reported accelerated implementation with a 300-tree random forest. Synthetic observations are sampled from the existing fold-specific augmentation pools; the generator is not retrained for each retained-data fraction.

### 6. PubMed concordance analysis

```bash
python scripts/run_pubmed_concordance.py \
  --ml_ranking outputs/extended_resampling/freq_top10_Ours_consensus_dedup.csv \
  --out_dir outputs/pubmed_concordance

python scripts/run_bibliometric_enrichment.py \
  --out_dir outputs/pubmed_concordance
```

The PubMed analysis requires internet access. Because PubMed is updated continuously, a new search may not reproduce the manuscript counts exactly. The frozen July 22, 2026 reference results are provided in `results/reference`.

## Reference values

The repository includes compact reference outputs for checking a rerun. Key manuscript values include:

- fused Top-3 selection frequency: ACC 92%, VTA 80%
- primary Top-3 validation AUC: 0.993 ± 0.033
- standardized baseline AUC: Graph-MDWGAN 0.970 ± 0.072; RealOnly 0.970 ± 0.086
- latest literature concordance: Spearman rho 0.536157; permutation P 0.022798

## Reproducibility notes

- Outer validation observations are not used for graph construction, generator training, checkpoint selection, feature ranking or scaling.
- Saline and methamphetamine generators share the model framework but use the group-specific optimization and loss-weighting branches implemented in `graph_mdwgan.py`.
- Figure 6 primary performance and Figure 7 standardized baseline performance come from different evaluation procedures and should not be compared as if they were the same estimate.
- The historical Figure 8 plotting script contained older hard-coded statistics and is not included here. The current concordance code calculates statistics directly from the latest output tables.

## Code provenance

The scripts were extracted from the final analysis archive used for the manuscript. File paths and import names were normalized for a GitHub repository; model definitions, numerical settings and statistical procedures were not rewritten.

## Citation

Please cite the associated manuscript when using this code. Bibliographic details can be added after publication.
