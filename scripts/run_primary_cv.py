# -*- coding: utf-8 -*-
"""
run_primary_cv.py

Leakage-controlled evaluation protocol:
Repeated stratified K-fold outer cross-validation.
Within each training fold: augmentation (Graph-MDWGAN/ROS/SMOTE), feature ranking
(feature_selection.py: fusion/consensus/GNN-lite), fixed-classifier fitting
(logistic regression), and held-out validation.

Outputs:
  cv_results/auc_k_long.csv
  cv_results/auc_k_mean_sd.csv
  cv_results/jaccard_mean_sd.csv
  cv_results/frequency_top10_{method}_{scheme}.csv
"""

import os
import re
import json
import math
import shutil
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Union

# ===== Local project dependencies =====
from graph_mdwgan import load_excel_data, train_group, postprocess_v10k, EPS_POS

META_COLS = {"Group", "Origin", "SampleID"}

# ----------------- Utility functions -----------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def feat_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS]

def y_from_group(series: pd.Series):
    return (series.astype(str).str.upper() == "METH").astype(int).to_numpy()

def safe_corrcoef(X_log: np.ndarray) -> np.ndarray:
    C = np.corrcoef(X_log.T).astype(np.float32)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return C

def dup_rates(real_np: np.ndarray, added_np: np.ndarray, decimals: int = 6):
    if added_np.shape[0] == 0:
        return float("nan"), float("nan")
    real_r = np.round(real_np.astype(np.float64), decimals)
    add_r  = np.round(added_np.astype(np.float64), decimals)
    real_set = set(map(tuple, real_r))
    add_tuples = list(map(tuple, add_r))
    dup_to_real = sum(1 for r in add_tuples if r in real_set) / (len(add_tuples) + 1e-12)
    dup_within  = 1.0 - (len(set(add_tuples)) / (len(add_tuples) + 1e-12))
    return float(dup_to_real), float(dup_within)

def _resolve_ckpt_path(best_info, ckpt_dir: Path, group: str) -> str:
    """Resolve a usable checkpoint path from train_group output or the checkpoint directory."""
    path = None
    if isinstance(best_info, dict):
        path = best_info.get("path") or best_info.get("ckpt")
    elif isinstance(best_info, str):
        path = best_info

    if path and os.path.exists(path):
        return path

    group_dir = ckpt_dir / group
    candidates = []
    if group_dir.exists():
        candidates.extend(sorted(group_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        if not candidates:
            candidates.extend(sorted(group_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))

    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        f"Unable to locate a checkpoint for {group}. train_group output={best_info}, ckpt_dir={ckpt_dir}"
    )

# ----------------- Graph-MDWGAN generator loading and sampling -----------------
def _load_generator_from_ckpt(ckpt_path: str, device: str):
    from graph_mdwgan import Generator
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("model_config", {})
    feat_dim = cfg.get("feat_dim")
    if feat_dim is None:
        raise ValueError("Checkpoint is missing feat_dim; the generator cannot be restored.")

    G = Generator(
        z_dim=cfg.get("z_dim", 128),
        feat_dim=feat_dim,
        hidden=cfg.get("g_hidden"),
        gm_width=cfg.get("gm_width"),
        dropout=cfg.get("dropout", 0.0)
    ).to(device)

    G.load_state_dict(ck["G"])
    G.eval()

    # Prefer exponential moving-average weights when available
    if "ema" in ck and isinstance(ck["ema"], dict):
        with torch.no_grad():
            for n, p in G.named_parameters():
                if p.requires_grad and (n in ck["ema"]):
                    arr = ck["ema"][n]
                    tensor = torch.tensor(arr, device=device, dtype=p.dtype)
                    p.data.copy_(tensor)

    Ahat = torch.tensor(ck["Ahat"], device=device, dtype=torch.float32)
    scaler_mean = torch.tensor(ck["scaler_mean"], device=device, dtype=torch.float32)
    scaler_scale = torch.tensor(ck["scaler_scale"], device=device, dtype=torch.float32)
    return G, Ahat, scaler_mean, scaler_scale

@torch.no_grad()
def _generate_synthetic_before(G, Ahat, scaler_mean, scaler_scale, n_samples: int, device: str, batch_size: int = 128):
    # Latent noise dimension
    if hasattr(G, "fc1") and hasattr(G.fc1, "in_features"):
        Z_DIM = G.fc1.in_features
    else:
        Z_DIM = 128

    out_list = []
    remain = int(n_samples)
    while remain > 0:
        b = min(batch_size, remain)
        z = torch.randn(b, Z_DIM, device=device)
        fk_std = G(z, Ahat)
        fk_log = fk_std * scaler_scale + scaler_mean
        fk_before = torch.expm1(fk_log).clamp_min(EPS_POS)
        out_list.append(fk_before.detach().cpu().numpy().astype(np.float32))
        remain -= b
    return np.vstack(out_list) if out_list else np.zeros((0, int(scaler_mean.shape[0])), dtype=np.float32)

# ----------------- ROS / SMOTE within-group augmentation -----------------
def make_ros_groupwise(df_real: pd.DataFrame, factor: int, seed: int):
    rng = np.random.RandomState(seed)
    out = [df_real.copy()]
    for g, sub in df_real.groupby("Group"):
        n_add = len(sub) * (factor - 1)
        add = sub.sample(n=n_add, replace=True, random_state=rng.randint(0, 10**9)).copy()
        add["Origin"] = "Synthetic"
        add["SampleID"] = [f"{g}_ROS_{i+1:05d}" for i in range(len(add))]
        out.append(add)
    df_all = pd.concat(out, ignore_index=True)
    df_added = df_all[df_all["SampleID"].astype(str).str.contains("_ROS_")].copy()
    return df_all, df_added

def smote_generate_group(X: np.ndarray, n_add: int, seed: int, k_neighbors: int = 5):
    if n_add <= 0:
        return np.zeros((0, X.shape[1]), float)
    n = X.shape[0]
    k_eff = min(k_neighbors, n - 1)
    if k_eff < 1:
        raise ValueError(f"SMOTE requires at least two samples per group; observed n={n}.")

    rng = np.random.RandomState(seed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xs)
    _, neigh = nn.kneighbors(Xs, return_distance=True)
    neigh = neigh[:, 1:]

    syn = np.empty((n_add, X.shape[1]), float)
    for t in range(n_add):
        i = rng.randint(0, n)
        j = rng.choice(neigh[i])
        lam = rng.rand()
        syn[t] = Xs[i] + lam * (Xs[j] - Xs[i])

    syn = scaler.inverse_transform(syn)
    syn = np.clip(syn, 0, None)
    return syn

def make_smote_groupwise(df_real: pd.DataFrame, factor: int, seed: int, k_neighbors: int = 5):
    feats = feat_cols(df_real)
    adds = []
    for gi, (g, sub) in enumerate(df_real.groupby("Group")):
        X = sub[feats].to_numpy(float)
        n_add = len(sub) * (factor - 1)
        Xsyn = smote_generate_group(X, n_add=n_add, seed=seed + 100 + gi, k_neighbors=k_neighbors)
        add = pd.DataFrame(Xsyn, columns=feats)
        add["Group"] = g
        add["Origin"] = "Synthetic"
        add["SampleID"] = [f"{g}_SMOTE_{i+1:05d}" for i in range(len(add))]
        adds.append(add)

    df_added = pd.concat(adds, ignore_index=True) if adds else pd.DataFrame(columns=df_real.columns)
    df_all = pd.concat([df_real, df_added[df_real.columns]], ignore_index=True)
    return df_all, df_added

# ----------------- Run feature ranking and load ranked outputs -----------------
def run_feature_selection(python_exe: str, fs_script: Path, csv_path: Path, out_dir: Path,
                          seed: int, topk_viz: int, gnn_k: int, gnn_weight: float, rf_trees: int, zthr: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for reproducible batch execution

    cmd = [
        python_exe, str(fs_script),
        "--csv", str(csv_path),
        "--use_origin", "Both",
        "--out", str(out_dir),
        "--topk", str(topk_viz),
        "--seed", str(seed),
        "--gnn_k", str(gnn_k),
        "--gnn_weight", str(gnn_weight),
        "--rf_trees", str(rf_trees),
        "--zthr", str(zthr),
        "--export_topn", "1,3,5,10,20",
        "--export_prefix", "cv_top"
    ]
    subprocess.check_call(cmd, env=env)

def read_rankings(out_dir: Path):
    # fusion
    fusion_path = out_dir / "gnn_results" / "consensus_plus_gnn.csv"
    df_f = pd.read_csv(fusion_path).sort_values("ConsensusPlusGNN", ascending=False)
    fusion_rank = df_f["BrainRegion"].astype(str).tolist()

    # consensus
    cons_path = out_dir / "consensus_ranking.csv"
    df_c = pd.read_csv(cons_path).sort_values("avg_rank", ascending=True)
    consensus_rank = df_c["BrainRegion"].astype(str).tolist()

    # gnn only
    gnn_path = out_dir / "gnn_results" / "gnn_lite_importance.csv"
    df_g = pd.read_csv(gnn_path).sort_values("GNNlite_coef_norm", ascending=False)
    gnn_rank = df_g["BrainRegion"].astype(str).tolist()

    return {
        "fusion": fusion_rank,
        "consensus": consensus_rank,
        "gnn": gnn_rank
    }

# ----------------- Fixed-classifier evaluation (logistic regression) -----------------
def eval_logreg_auc(train_df: pd.DataFrame, val_df: pd.DataFrame, rois: list[str], seed: int):
    Xtr = train_df[rois].to_numpy(float)
    ytr = y_from_group(train_df["Group"])
    Xva = val_df[rois].to_numpy(float)
    yva = y_from_group(val_df["Group"])

    # AUC is undefined when the validation fold contains only one class
    if len(np.unique(yva)) < 2:
        return float("nan"), float("nan")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=10000, class_weight="balanced",
        random_state=seed
    )
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xva)[:, 1]

    auc = float(roc_auc_score(yva, prob))
    auprc = float(average_precision_score(yva, prob))
    return auc, auprc

# ----------------- Ranking stability: Jaccard similarity -----------------
def jaccard(a: set, b: set):
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u

def pairwise_jaccard_mean_sd(list_of_sets: list[set]):
    n = len(list_of_sets)
    if n < 2:
        return 0.0, 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(list_of_sets[i], list_of_sets[j]))
    vals = np.array(vals, float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

# ----------------- Main program -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, default="data/brain_region_cfos.xlsx")
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--factor", type=int, default=8)

    # Cross-validation settings
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=20250829)

    # Augmentation methods
    ap.add_argument("--methods", type=str, default="Ours,ROS,SMOTE")  # Specify a subset such as "ROS,SMOTE" for a reduced run
    ap.add_argument("--strict_ours", action="store_true",
                    help="Retrain the Graph-MDWGAN generator independently within every outer fold. This is computationally intensive but provides the strictest leakage control.")

    # Feature-selection parameters
    ap.add_argument("--fs_script", type=str, default="scripts/feature_selection.py")
    ap.add_argument("--topk_viz", type=int, default=20)
    ap.add_argument("--gnn_k", type=int, default=5)
    ap.add_argument("--gnn_weight", type=float, default=1.0/6.0)
    ap.add_argument("--rf_trees", type=int, default=500)
    ap.add_argument("--zthr", type=float, default=0.0)

    # SMOTE parameters
    ap.add_argument("--smote_k", type=int, default=5)

    # Output directory
    ap.add_argument("--out_dir", type=str, default="outputs/cv_results")
    ap.add_argument("--keep_fold_artifacts", action="store_true",
                    help="Retain fold-level feature-ranking outputs in addition to summary files.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_script = Path(args.fs_script)
    if not fs_script.exists():
        raise FileNotFoundError(f"Feature-selection script not found: {fs_script}. Place this script and feature_selection.py in the same directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.cv_seed)

    # 1) Load the full dataset (rows=samples, columns=ROIs)
    Xs, Xm, roi_names = load_excel_data(args.excel, args.sheet)
    nS, nM = Xs.shape[0], Xm.shape[0]
    p = len(roi_names)

    # Construct the full real-sample DataFrame used for outer CV splitting
    rows: list[Dict[str, Union[str, float]]] = []
    for i in range(nS):
        r: Dict[str, Union[str, float]] = {"Group": "Saline", "Origin": "RealTrain", "SampleID": f"Saline_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            r[str(roi)] = float(Xs[i, j])
        rows.append(r)
    for i in range(nM):
        r: Dict[str, Union[str, float]] = {"Group": "METH", "Origin": "RealTrain", "SampleID": f"METH_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            r[str(roi)] = float(Xm[i, j])
        rows.append(r)

    df_all = pd.DataFrame(rows)
    feats = feat_cols(df_all)
    y_all = y_from_group(df_all["Group"])

    # 2) Repeated Stratified KFold
    rskf = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    k_list = [1, 3, 5, 10, 20]
    k_list = [k for k in k_list if k <= p]

    # Store performance metrics and top-k sets for stability analysis
    auc_rows = []
    topk_sets = {}  # key=(method, scheme, k) -> list[set]

    # 3) Iterate over outer folds
    fold_id = 0
    python_exe = sys.executable

    for tr_idx, va_idx in rskf.split(df_all[feats].to_numpy(float), y_all):
        fold_id += 1
        fold_seed = args.cv_seed + fold_id * 17
        set_seed(fold_seed)

        df_tr_real = df_all.iloc[tr_idx].copy()
        df_va = df_all.iloc[va_idx].copy()
        df_tr_real["Origin"] = "RealTrain"

        # Estimate Ahat from the training fold only to prevent information leakage
        X_train_all = df_tr_real[feats].to_numpy(float)
        X_train_log = np.log1p(np.clip(X_train_all, 0, None))
        Ahat_np = safe_corrcoef(X_train_log)

        # Fold-specific working directory
        fold_dir = out_dir / f"fold_{fold_id:04d}"
        tmp_dir = fold_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for method in methods:
            method_name = method

            # --------- 3.1 Generate the augmented training set for this fold ---------
            if method_name.upper() == "OURS":
                # Strict mode: retrain the generator independently within each fold
                # Standard mode: training remains fold-specific; checkpoint reuse can be implemented separately if required
                ckpt_dir = fold_dir / "ckpt_ours"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                # Split the training fold by experimental group
                Xs_tr = df_tr_real[df_tr_real["Group"] == "Saline"][feats].to_numpy(float)
                Xm_tr = df_tr_real[df_tr_real["Group"] == "METH"][feats].to_numpy(float)

                # Retrain the generator to maintain fold independence
                best_S, _ = train_group("Saline", Xs_tr, Ahat_np, seed=fold_seed, out_dir=str(ckpt_dir), force_retrain=args.strict_ours)
                best_M, _ = train_group("METH",   Xm_tr, Ahat_np, seed=fold_seed + 1, out_dir=str(ckpt_dir), force_retrain=args.strict_ours)

                ckpt_S = _resolve_ckpt_path(best_S, ckpt_dir, "Saline")
                ckpt_M = _resolve_ckpt_path(best_M, ckpt_dir, "METH")

                GS, AhatS, meanS, scaleS = _load_generator_from_ckpt(ckpt_S, device=device)
                GM, AhatM, meanM, scaleM = _load_generator_from_ckpt(ckpt_M, device=device)

                n_add_S = Xs_tr.shape[0] * (args.factor - 1)
                n_add_M = Xm_tr.shape[0] * (args.factor - 1)

                syn_S = _generate_synthetic_before(GS, AhatS, meanS, scaleS, n_add_S, device=device)
                syn_M = _generate_synthetic_before(GM, AhatM, meanM, scaleM, n_add_M, device=device)

                df_syn_S = pd.DataFrame(syn_S, columns=feats)
                df_syn_S["Group"] = "Saline"
                df_syn_S["Origin"] = "Synthetic"
                df_syn_S["SampleID"] = [f"Saline_Ours_{i+1:05d}" for i in range(len(df_syn_S))]

                df_syn_M = pd.DataFrame(syn_M, columns=feats)
                df_syn_M["Group"] = "METH"
                df_syn_M["Origin"] = "Synthetic"
                df_syn_M["SampleID"] = [f"METH_Ours_{i+1:05d}" for i in range(len(df_syn_M))]

                df_tr_aug = pd.concat([df_tr_real, df_syn_S[df_tr_real.columns], df_syn_M[df_tr_real.columns]],
                                      ignore_index=True)

                # Record duplicate rates for audit and sensitivity reporting
                dtr_S, dwi_S = dup_rates(Xs_tr, syn_S)
                dtr_M, dwi_M = dup_rates(Xm_tr, syn_M)
                dup_to_real = (dtr_S * len(Xs_tr) + dtr_M * len(Xm_tr)) / (len(Xs_tr) + len(Xm_tr) + 1e-12)
                dup_within  = (dwi_S * len(Xs_tr) + dwi_M * len(Xm_tr)) / (len(Xs_tr) + len(Xm_tr) + 1e-12)

            elif method_name.upper() == "ROS":
                df_tr_aug, df_added = make_ros_groupwise(df_tr_real, args.factor, seed=fold_seed + 3)
                X_real = df_tr_real[feats].to_numpy(float)
                X_add  = df_added[feats].to_numpy(float)
                dup_to_real, dup_within = dup_rates(X_real, X_add)

            elif method_name.upper() == "SMOTE":
                df_tr_aug, df_added = make_smote_groupwise(df_tr_real, args.factor, seed=fold_seed + 5, k_neighbors=args.smote_k)
                X_real = df_tr_real[feats].to_numpy(float)
                X_add  = df_added[feats].to_numpy(float)
                dup_to_real, dup_within = dup_rates(X_real, X_add)

            else:
                raise ValueError(f"Unknown method: {method_name}. Supported methods are Ours, ROS, and SMOTE.")

            # --------- 3.2 Run feature ranking using augmented training-fold data only ---------
            method_tag = method_name.upper()
            csv_path = tmp_dir / f"train_{method_tag}_x{args.factor}.csv"
            df_tr_aug.to_csv(csv_path, index=False, encoding="utf-8-sig")

            fs_out = fold_dir / f"fs_{method_tag}"
            need_fusion = fs_out / "gnn_results" / "consensus_plus_gnn.csv"
            if not need_fusion.exists():
                run_feature_selection(
                    python_exe=python_exe,
                    fs_script=fs_script,
                    csv_path=csv_path,
                    out_dir=fs_out,
                    seed=fold_seed,
                    topk_viz=args.topk_viz,
                    gnn_k=args.gnn_k,
                    gnn_weight=args.gnn_weight,
                    rf_trees=args.rf_trees,
                    zthr=args.zthr
                )

            ranks = read_rankings(fs_out)

            # --------- 3.3 Evaluate top-k performance on the held-out validation fold ---------
            for scheme, rank_list in ranks.items():
                for k in k_list:
                    rois = rank_list[:k]
                    auc, auprc = eval_logreg_auc(df_tr_aug, df_va, rois, seed=fold_seed)

                    auc_rows.append({
                        "fold": fold_id,
                        "seed": fold_seed,
                        "method": method_name,
                        "scheme": scheme,
                        "k": k,
                        "auc": auc,
                        "auprc": auprc,
                        "dup_to_real": float(dup_to_real),
                        "dup_within": float(dup_within)
                    })

                    key = (method_name, scheme, k)
                    topk_sets.setdefault(key, []).append(set(rois))

        # Remove intermediate artifacts unless fold-level outputs are requested
        if not args.keep_fold_artifacts:
            # Remove temporary files while retaining requested feature-ranking outputs
            # Checkpoints and ranking outputs are retained when traceability is required.
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # 4) Write the long-format performance table
    df_long = pd.DataFrame(auc_rows)
    df_long.to_csv(out_dir / "auc_k_long.csv", index=False, encoding="utf-8-sig")

    # 5) Summarize performance as mean and standard deviation
    df_sum = (df_long
              .groupby(["method", "scheme", "k"])
              .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                   auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
                   dup_to_real_mean=("dup_to_real", "mean"), dup_to_real_sd=("dup_to_real", "std"),
                   dup_within_mean=("dup_within", "mean"), dup_within_sd=("dup_within", "std"))
              .reset_index())
    df_sum.to_csv(out_dir / "auc_k_mean_sd.csv", index=False, encoding="utf-8-sig")

    # 6) Summarize Jaccard stability across folds
    jac_rows = []
    for (method, scheme, k), sets_list in topk_sets.items():
        jm, jsd = pairwise_jaccard_mean_sd(sets_list)
        jac_rows.append({
            "method": method,
            "scheme": scheme,
            "k": k,
            "jaccard_mean": jm,
            "jaccard_sd": jsd,
            "n_runs": len(sets_list)
        })
    df_jac = pd.DataFrame(jac_rows).sort_values(["method", "scheme", "k"])
    df_jac.to_csv(out_dir / "jaccard_mean_sd.csv", index=False, encoding="utf-8-sig")

    # 7) Summarize top-10 selection frequencies (or all ROIs when fewer than 10)
    k_freq = 10 if 10 in k_list else max(k_list)
    for method in sorted(set(df_long["method"].tolist())):
        for scheme in sorted(set(df_long["scheme"].tolist())):
            key = (method, scheme, k_freq)
            if key not in topk_sets:
                continue
            sets_list = topk_sets[key]
            freq = {}
            for s in sets_list:
                for r in s:
                    freq[r] = freq.get(r, 0) + 1
            df_freq = pd.DataFrame({
                "BrainRegion": list(freq.keys()),
                "freq": [freq[r] for r in freq.keys()],
                "freq_rate": [freq[r] / len(sets_list) for r in freq.keys()]
            }).sort_values("freq_rate", ascending=False)
            df_freq.to_csv(out_dir / f"frequency_top{k_freq}_{method}_{scheme}.csv",
                           index=False, encoding="utf-8-sig")

    print("\nDone.")
    print("Saved:")
    print("  ", out_dir / "auc_k_long.csv")
    print("  ", out_dir / "auc_k_mean_sd.csv")
    print("  ", out_dir / "jaccard_mean_sd.csv")
    print("  ", out_dir / f"frequency_top{k_freq}_<method>_<scheme>.csv")

if __name__ == "__main__":
    main()
