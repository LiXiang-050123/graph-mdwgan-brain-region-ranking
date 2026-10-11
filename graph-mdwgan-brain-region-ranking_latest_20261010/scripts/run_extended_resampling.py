# -*- coding: utf-8 -*-
"""
onezone_roi_suite_reuse_ckpt.py

一区SCI：脑区筛选稳健性/公平性/负对照/Oracle 全套评估
并【复用旧脚本 cv_roi_pipeline_repeatedkfold_x8.py 的 Ours ckpt（五折记忆）】：
- 同样的 RepeatedStratifiedKFold(random_state=cv_seed, n_splits, n_repeats)
- 同样的 fold_seed = cv_seed + fold_id * 17
- 同样的 fold_dir = reuse_cv_dir/fold_{fold_id:04d}/ckpt_ours/(Saline|METH)

输出目录（默认 outputs/extended_resampling）：
  perf_long.csv / perf_mean_sd.csv
  stability_long.csv / stability_mean_sd.csv
  freq_top10_<method>_<scheme>_<fair>.csv
  random_roi_baseline.csv
  oracle_long.csv / oracle_summary.csv

说明：
1) 复用 ckpt 仅减少“训练生成器”的耗时；每 fold 仍会重新采样生成合成数据（很快）。
2) 稳定性选择（bootstrap）需要重新运行（这是你补齐一区证据链的核心）。
"""

import os
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --------- 全局警告抑制 ---------
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DataConversionWarning)
except Exception:
    pass

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss
)
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata


# ===== 依赖你的工程文件（同目录）=====
try:
    from .graph_mdwgan import EPS_POS, load_excel_data, train_group
    from .ranking_utils import BASE_RANKING_METHODS, consensus_and_fusion
except ImportError:
    from graph_mdwgan import EPS_POS, load_excel_data, train_group
    from ranking_utils import BASE_RANKING_METHODS, consensus_and_fusion

META_COLS = {"Group", "Origin", "SampleID"}


# ----------------- 基础工具 -----------------
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

def round_tuples(X: np.ndarray, decimals: int = 6):
    Xr = np.round(X.astype(np.float64), decimals)
    return list(map(tuple, Xr))

def dup_rates(real_np: np.ndarray, added_np: np.ndarray, decimals: int = 6):
    if added_np.shape[0] == 0:
        return float("nan"), float("nan")
    real_set = set(round_tuples(real_np, decimals))
    add_t = round_tuples(added_np, decimals)
    dup_to_real = sum(1 for t in add_t if t in real_set) / (len(add_t) + 1e-12)
    dup_within  = 1.0 - (len(set(add_t)) / (len(add_t) + 1e-12))
    return float(dup_to_real), float(dup_within)

def make_sample_weights_by_uniques(X: np.ndarray, decimals: int = 6, target_total_weight: float | None = None):
    tuples = round_tuples(X, decimals)
    counts = {}
    for t in tuples:
        counts[t] = counts.get(t, 0) + 1
    w = np.array([1.0 / counts[t] for t in tuples], dtype=np.float64)
    if target_total_weight is not None and w.sum() > 0:
        w = w * (float(target_total_weight) / w.sum())
    return w

def dedup_against_real(df_real: pd.DataFrame, df_syn: pd.DataFrame, decimals: int = 6):
    feats = feat_cols(df_real)
    real_set = set(round_tuples(df_real[feats].to_numpy(float), decimals))

    syn = df_syn.copy()
    syn_t = round_tuples(syn[feats].to_numpy(float), decimals)
    keep_mask = np.array([t not in real_set for t in syn_t], dtype=bool)
    syn = syn.loc[keep_mask].copy()

    syn["_k"] = round_tuples(syn[feats].to_numpy(float), decimals)
    syn = syn.drop_duplicates(subset=["_k"]).drop(columns=["_k"])
    return syn


# ----------------- 复用 ckpt：定位 + 加载 + 采样 -----------------
def _resolve_ckpt_path(ckpt_dir: Path, group: str) -> str:
    """
    按旧脚本结构：fold_xxxx/ckpt_ours/<group>/best*.pt
    找最新的 best*.pt；若没有 best，则找任何 *.pt
    """
    group_dir = ckpt_dir / group
    if not group_dir.exists():
        raise FileNotFoundError(f"找不到 {group_dir}")

    candidates = sorted(group_dir.glob("best_seed*.pt"))

    if not candidates:
        raise FileNotFoundError(f"{group_dir} 下没有任何 .pt ckpt 文件")
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one final best_seed*.pt under {group_dir}, "
            f"found {len(candidates)}. Refusing to select a checkpoint by mtime."
        )
    return str(candidates[0])

def _load_generator_from_ckpt(ckpt_path: str, device: str):
    from fifth_gen_integrated import Generator
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("model_config", {})
    feat_dim = cfg.get("feat_dim")
    if feat_dim is None:
        raise ValueError("ckpt 缺少 feat_dim，无法恢复生成器。")

    G = Generator(
        z_dim=cfg.get("z_dim", 128),
        feat_dim=feat_dim,
        hidden=cfg.get("g_hidden"),
        gm_width=cfg.get("gm_width"),
        dropout=cfg.get("dropout", 0.0)
    ).to(device)

    G.load_state_dict(ck["G"])
    G.eval()

    # EMA 优先
    if "ema" in ck and isinstance(ck["ema"], dict):
        with torch.no_grad():
            for n, p in G.named_parameters():
                if p.requires_grad and (n in ck["ema"]):
                    p.data.copy_(torch.tensor(ck["ema"][n], device=device, dtype=p.dtype))

    Ahat = torch.tensor(ck["Ahat"], device=device, dtype=torch.float32)
    scaler_mean = torch.tensor(ck["scaler_mean"], device=device, dtype=torch.float32)
    scaler_scale = torch.tensor(ck["scaler_scale"], device=device, dtype=torch.float32)
    return G, Ahat, scaler_mean, scaler_scale

@torch.no_grad()
def _generate_synthetic_before(G, Ahat, scaler_mean, scaler_scale, n_samples: int, device: str, batch_size: int = 128):
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

def make_ours_groupwise_reuse_ckpt(df_tr_real: pd.DataFrame, factor: int, fold_seed: int,
                                  fold_dir_old: Path, strict_retrain: bool):
    """
    优先复用 fold_dir_old/ckpt_ours 的 ckpt。
    若 ckpt 缺失或 strict_retrain=True，则按旧脚本逻辑在该 fold 训练生成器并保存到同目录。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = feat_cols(df_tr_real)

    ckpt_dir = fold_dir_old / "ckpt_ours"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 训练fold内 Ahat（严格避免泄漏）
    X_train_all = df_tr_real[feats].to_numpy(float)
    X_train_log = np.log1p(np.clip(X_train_all, 0, None))
    Ahat_np = safe_corrcoef(X_train_log)

    dfS = df_tr_real[df_tr_real["Group"] == "Saline"]
    dfM = df_tr_real[df_tr_real["Group"] == "METH"]
    XS = dfS[feats].to_numpy(float)
    XM = dfM[feats].to_numpy(float)

    # 1) 尝试复用 ckpt
    ckpt_S = ckpt_M = None
    if (not strict_retrain):
        try:
            ckpt_S = _resolve_ckpt_path(ckpt_dir, "Saline")
            ckpt_M = _resolve_ckpt_path(ckpt_dir, "METH")
        except Exception:
            ckpt_S = ckpt_M = None

    # 2) 如无法复用，则训练并产生 ckpt（写入旧fold目录，便于下次复用）
    if ckpt_S is None or ckpt_M is None:
        print(f"[Ours][fold_seed={fold_seed}] 未找到可复用ckpt，开始训练并写入: {ckpt_dir}")
        best_S, _ = train_group("Saline", XS, Ahat_np, seed=fold_seed,   out_dir=str(ckpt_dir), force_retrain=True)
        best_M, _ = train_group("METH",   XM, Ahat_np, seed=fold_seed+1, out_dir=str(ckpt_dir), force_retrain=True)
        # 训练后按目录定位最新 best*.pt
        ckpt_S = _resolve_ckpt_path(ckpt_dir, "Saline")
        ckpt_M = _resolve_ckpt_path(ckpt_dir, "METH")
        print(f"[Ours][fold_seed={fold_seed}] 新训练ckpt：Saline={ckpt_S} | METH={ckpt_M}")
    else:
        print(f"[Ours][fold_seed={fold_seed}] 复用ckpt：Saline={ckpt_S} | METH={ckpt_M}")

    GS, AhatS, meanS, scaleS = _load_generator_from_ckpt(ckpt_S, device=device)
    GM, AhatM, meanM, scaleM = _load_generator_from_ckpt(ckpt_M, device=device)

    n_add_S = XS.shape[0] * (factor - 1)
    n_add_M = XM.shape[0] * (factor - 1)

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
    df_added = pd.concat([df_syn_S[df_tr_real.columns], df_syn_M[df_tr_real.columns]], ignore_index=True)

    dtr_S, dwi_S = dup_rates(XS, syn_S)
    dtr_M, dwi_M = dup_rates(XM, syn_M)
    dup_to_real = (dtr_S * len(XS) + dtr_M * len(XM)) / (len(XS) + len(XM) + 1e-12)
    dup_within  = (dwi_S * len(XS) + dwi_M * len(XM)) / (len(XS) + len(XM) + 1e-12)

    return df_tr_aug, df_added, dup_to_real, dup_within


# ----------------- ROS / SMOTE -----------------
def make_ros_groupwise(df_real: pd.DataFrame, factor: int, seed: int):
    rng = np.random.RandomState(seed)
    out = [df_real.copy()]
    adds = []
    for g, sub in df_real.groupby("Group"):
        n_add = len(sub) * (factor - 1)
        add = sub.sample(n=n_add, replace=True, random_state=rng.randint(0, 10**9)).copy()
        add["Origin"] = "Synthetic"
        add["SampleID"] = [f"{g}_ROS_{i+1:05d}" for i in range(len(add))]
        adds.append(add)
        out.append(add)
    df_all = pd.concat(out, ignore_index=True)
    df_added = pd.concat(adds, ignore_index=True) if adds else pd.DataFrame(columns=df_real.columns)
    return df_all, df_added

def smote_generate_group(X: np.ndarray, n_add: int, seed: int, k_neighbors: int = 5):
    if n_add <= 0:
        return np.zeros((0, X.shape[1]), float)
    n = X.shape[0]
    k_eff = min(k_neighbors, n - 1)
    if k_eff < 1:
        return np.zeros((0, X.shape[1]), float)

    rng = np.random.RandomState(seed)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xs)
    neigh = np.asarray(nn.kneighbors(Xs, return_distance=False))[:, 1:]

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


# ----------------- 轻量筛选（bootstrap 反复跑用） -----------------
def rank_anova(X, y):
    f, _ = f_classif(X, y)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

def rank_mi(X, y, seed=0):
    mi = mutual_info_classif(X, y, random_state=seed, discrete_features=False)
    return np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)

def rank_lasso(X, y, seed=0):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(
        penalty="l1", solver="saga", C=1.0, max_iter=10000,
        class_weight="balanced", random_state=seed
    )
    clf.fit(Xs, y)
    return np.abs(clf.coef_.ravel())

def rank_rf(X, y, seed=0):
    clf = RandomForestClassifier(
        n_estimators=400, random_state=seed, class_weight="balanced_subsample",
        n_jobs=1
    )
    clf.fit(X, y)
    return np.nan_to_num(clf.feature_importances_, nan=0.0)

def rank_rfe_svm(X, y, seed=0):
    Xs = StandardScaler().fit_transform(X)
    svc = SVC(kernel="linear", class_weight="balanced", random_state=seed)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(Xs, y)
    score = (rfe.ranking_.max() - rfe.ranking_ + 1).astype(float)
    return score

def rank_gnn_lite_proxy(X, y, seed=0):
    Xc = np.log1p(np.clip(X, 0, None))
    C = np.abs(np.corrcoef(Xc.T))
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    row_sum = C.sum(axis=1, keepdims=True) + 1e-12
    A = C / row_sum
    Xs = Xc @ A
    Xz = StandardScaler().fit_transform(Xs)
    clf = LogisticRegression(
        penalty="l2", solver="lbfgs", C=1.0, max_iter=10000,
        class_weight="balanced", random_state=seed
    )
    clf.fit(Xz, y)
    return np.abs(clf.coef_.ravel())

def bootstrap_stability(train_df, feats, seed, B=50, k_list=(5,10,20), gnn_weight=1/6):
    rng = np.random.RandomState(seed)
    y = y_from_group(train_df["Group"])
    X = train_df[feats].to_numpy(float)

    topk_sets = {(scheme,k): [] for scheme in ["consensus","fusion","gnn"] for k in k_list}
    freq = {scheme: {} for scheme in ["consensus","fusion","gnn"]}

    n = len(train_df)
    for b in tqdm(range(B), desc="Bootstrap", leave=False):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        Xb = X[idx]; yb = y[idx]

        base_scores = {
            "ANOVA": rank_anova(Xb, yb),
            "MI":    rank_mi(Xb, yb, seed=seed + 10 + b),
            "LASSO": rank_lasso(Xb, yb, seed=seed + 20 + b),
            "RF":    rank_rf(Xb, yb, seed=seed + 30 + b),
            "RFE":   rank_rfe_svm(Xb, yb, seed=seed + 40 + b),
        }
        gnn_score = rank_gnn_lite_proxy(Xb, yb, seed=seed + 50 + b)
        cons_score, fus_score = consensus_and_fusion(
            base_scores,
            gnn_score=gnn_score,
            gnn_weight=gnn_weight,
        )

        def to_rank(score):
            order = np.argsort(-score)
            return [feats[i] for i in order]

        cons_rank = to_rank(cons_score)
        fus_rank  = to_rank(fus_score)
        gnn_rank  = to_rank(gnn_score)

        for k in k_list:
            topk_sets[("consensus",k)].append(set(cons_rank[:k]))
            topk_sets[("fusion",k)].append(set(fus_rank[:k]))
            topk_sets[("gnn",k)].append(set(gnn_rank[:k]))

        # 频率（默认用 Top10）
        kfreq = 10 if 10 in k_list else max(k_list)
        for roi in fus_rank[:kfreq]:
            freq["fusion"][roi] = freq["fusion"].get(roi, 0) + 1
        for roi in cons_rank[:kfreq]:
            freq["consensus"][roi] = freq["consensus"].get(roi, 0) + 1
        for roi in gnn_rank[:kfreq]:
            freq["gnn"][roi] = freq["gnn"].get(roi, 0) + 1

    return topk_sets, freq

def _bootstrap_cache_path(cache_dir: Path, fold_id: int, method: str, fair_tag: str,
                          seed: int, B: int, k_list, gnn_weight: float) -> Path:
    k_str = "-".join(str(k) for k in k_list)
    fname = f"fold{fold_id:04d}__{method}__{fair_tag}__seed{seed}__B{B}__k{k_str}__gw{gnn_weight:.6f}.pkl"
    return cache_dir / fname

def load_bootstrap_cache(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_bootstrap_cache(path: Path, topk_sets, freq):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"topk_sets": topk_sets, "freq": freq}, f, protocol=pickle.HIGHEST_PROTOCOL)

def jaccard(a: set, b: set):
    u = len(a | b)
    return 0.0 if u == 0 else len(a & b) / u

def pairwise_jaccard_mean_sd(list_of_sets):
    n = len(list_of_sets)
    if n < 2:
        return 0.0, 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(list_of_sets[i], list_of_sets[j]))
    vals = np.array(vals, float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


# ----------------- 评估模型（固定LogReg） -----------------
def eval_metrics(train_df, val_df, rois, seed=0, sample_weight=None):
    Xtr = train_df[rois].to_numpy(float)
    ytr = y_from_group(train_df["Group"])
    Xva = val_df[rois].to_numpy(float)
    yva = y_from_group(val_df["Group"])

    if len(np.unique(yva)) < 2:
        return {"auc": np.nan, "auprc": np.nan, "mcc": np.nan, "balacc": np.nan, "logloss": np.nan, "brier": np.nan}

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    clf = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=10000,
        class_weight="balanced", random_state=seed
    )
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    prob = clf.predict_proba(Xva)[:, 1]
    pred = (prob >= 0.5).astype(int)

    out = {}
    out["auc"] = float(roc_auc_score(yva, prob))
    out["auprc"] = float(average_precision_score(yva, prob))
    out["mcc"] = float(matthews_corrcoef(yva, pred))
    out["balacc"] = float(balanced_accuracy_score(yva, pred))
    out["logloss"] = float(log_loss(yva, np.vstack([1-prob, prob]).T, labels=[0,1]))
    out["brier"] = float(np.mean((prob - yva) ** 2))
    return out


# ----------------- Oracle：内层CV自动挑最佳单法+k -----------------
def oracle_select(train_df, feats, seed, k_list=(3,5,10), inner_splits=3):
    X = train_df[feats].to_numpy(float)
    y = y_from_group(train_df["Group"])
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)

    methods = ["ANOVA","MI","LASSO","RF","RFE","GNN"]
    best = {"method": None, "k": None, "score": -1e9}

    for m in methods:
        for k in k_list:
            scores = []
            for tr_idx, va_idx in inner.split(X, y):
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]
                if len(np.unique(yva)) < 2:
                    continue

                if m == "ANOVA": s = rank_anova(Xtr, ytr)
                elif m == "MI":  s = rank_mi(Xtr, ytr, seed=seed+1)
                elif m == "LASSO": s = rank_lasso(Xtr, ytr, seed=seed+2)
                elif m == "RF": s = rank_rf(Xtr, ytr, seed=seed+3)
                elif m == "RFE": s = rank_rfe_svm(Xtr, ytr, seed=seed+4)
                elif m == "GNN": s = rank_gnn_lite_proxy(Xtr, ytr, seed=seed+5)
                else: continue

                order = np.argsort(-s)
                rois = [feats[i] for i in order[:k]]
                df_tr = train_df.iloc[tr_idx].copy()
                df_va = train_df.iloc[va_idx].copy()
                met = eval_metrics(df_tr, df_va, rois, seed=seed)
                scores.append(met["auprc"])

            if scores:
                sc = float(np.mean(scores))
                if sc > best["score"]:
                    best = {"method": m, "k": k, "score": sc}
    return best


def main():
    ap = argparse.ArgumentParser()

    # 数据
    ap.add_argument("--excel", type=str, default="data/brain_region_cfos.xlsx")
    ap.add_argument("--sheet", type=str, default=None)

    # 与旧脚本一致的 CV 参数
    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=20250829)

    # 方法
    ap.add_argument("--methods", type=str, default="Ours,ROS,SMOTE")
    ap.add_argument("--smote_k", type=int, default=5)

    # 复用旧 fold 目录（必须指向你旧脚本的 out_dir，比如 cv_results）
    ap.add_argument("--reuse_cv_dir", type=str, default="outputs/cv_results",
                    help="旧脚本输出目录（包含 fold_0001/ckpt_ours/...），用于复用Ours ckpt")

    # Ours 控制
    ap.add_argument("--strict_ours", action="store_true",
                    help="强制重训生成器（不复用记忆）。默认不加：优先复用旧ckpt。")

    # 稳定性选择
    ap.add_argument("--k_list", type=str, default="1,3,5,10,20")
    ap.add_argument("--bootstrap_B", type=int, default=50)
    ap.add_argument("--gnn_weight", type=float, default=1.0/6.0)

    # 公平对照（顶级审稿人标准：两项都开）
    ap.add_argument("--dedup_decimals", type=int, default=6)

    # Random ROI 基线
    ap.add_argument("--random_roi_reps", type=int, default=100)

    # Oracle
    ap.add_argument("--oracle_k_list", type=str, default="3,5,10")

    # 输出
    ap.add_argument("--out_dir", type=str, default="outputs/extended_resampling")
    ap.add_argument("--cache_bootstrap", action="store_true",
                    help="启用bootstrap缓存（同一fold/method/fair参数重复运行时直接读取）")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="bootstrap缓存目录（默认：out_dir/bootstrap_cache）")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_dir / "bootstrap_cache")

    reuse_dir = Path(args.reuse_cv_dir)
    if not reuse_dir.exists():
        raise FileNotFoundError(f"找不到 reuse_cv_dir：{reuse_dir}（请把旧的 cv_results 目录放在同级或给绝对路径）")

    set_seed(args.cv_seed)

    # 读数据（构造顺序必须与旧脚本一致：先Saline后METH）
    Xs, Xm, roi_names = load_excel_data(args.excel, args.sheet)
    rows: list[dict[str, object]] = []
    for i in range(Xs.shape[0]):
        row: dict[str, object] = {"Group":"Saline","Origin":"RealTrain","SampleID":f"Saline_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            row[str(roi)] = float(Xs[i,j])
        rows.append(row)
    for i in range(Xm.shape[0]):
        row: dict[str, object] = {"Group":"METH","Origin":"RealTrain","SampleID":f"METH_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            row[str(roi)] = float(Xm[i,j])
        rows.append(row)

    df_all = pd.DataFrame(rows)
    feats = feat_cols(df_all)
    y_all = y_from_group(df_all["Group"])

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    k_list = [k for k in k_list if 1 <= k <= len(feats)]
    oracle_k_list = [int(x) for x in args.oracle_k_list.split(",") if x.strip()]
    oracle_k_list = [k for k in oracle_k_list if 1 <= k <= len(feats)]

    rskf = RepeatedStratifiedKFold(
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.cv_seed
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    perf_rows = []
    stab_rows = []
    rand_rows = []
    oracle_rows = []

    # 频率累计（跨 fold×bootstrap）
    freq_acc = {}   # key=(method,scheme,fair)-> dict roi->count
    freq_total = {} # key-> total_runs

    total_folds = args.n_splits * args.n_repeats
    for fold_id, (tr_idx, va_idx) in enumerate(
        tqdm(rskf.split(df_all[feats].to_numpy(float), y_all), total=total_folds, desc="CV Folds"),
        start=1
    ):

        # 【关键对齐】：与旧脚本一致
        fold_seed = args.cv_seed + fold_id * 17
        set_seed(fold_seed)

        df_tr_real = df_all.iloc[tr_idx].copy()
        df_va = df_all.iloc[va_idx].copy()
        df_tr_real["Origin"] = "RealTrain"

        # 【关键对齐】：旧脚本 fold 目录
        fold_dir_old = reuse_dir / f"fold_{fold_id:04d}"

        for method in tqdm(methods, desc="Methods", leave=False):
            mtag = method.upper()

            # 1) 生成扩增训练集（Ours：优先复用旧ckpt）
            if mtag == "OURS":
                df_tr_aug, df_added, dup_to_real, dup_within = make_ours_groupwise_reuse_ckpt(
                    df_tr_real, args.factor, fold_seed, fold_dir_old, strict_retrain=args.strict_ours
                )
            elif mtag == "ROS":
                df_tr_aug, df_added = make_ros_groupwise(df_tr_real, args.factor, seed=fold_seed + 3)
                X_real = df_tr_real[feats].to_numpy(float)
                X_add  = df_added[feats].to_numpy(float)
                dup_to_real, dup_within = dup_rates(X_real, X_add, decimals=args.dedup_decimals)
            elif mtag == "SMOTE":
                df_tr_aug, df_added = make_smote_groupwise(df_tr_real, args.factor, seed=fold_seed + 5, k_neighbors=args.smote_k)
                X_real = df_tr_real[feats].to_numpy(float)
                X_add  = df_added[feats].to_numpy(float)
                dup_to_real, dup_within = dup_rates(X_real, X_add, decimals=args.dedup_decimals)
            else:
                raise ValueError(f"未知方法：{method}")

            # 训练集版本：raw / dedup / fixunique（两项都做）
            target_total_weight = float(len(df_tr_real) * args.factor)
            train_versions: list[tuple[str, pd.DataFrame, np.ndarray | None]] = [("raw", df_tr_aug, None)]

            # dedup
            df_syn = df_tr_aug[df_tr_aug["Origin"] != "RealTrain"].copy()
            df_syn_d = dedup_against_real(df_tr_real, df_syn, decimals=args.dedup_decimals)
            df_dedup = pd.concat([df_tr_real, df_syn_d[df_tr_real.columns]], ignore_index=True)
            train_versions.append(("dedup", df_dedup, None))

            # fixunique（对 raw 的组合数据赋权）
            X_comb = df_tr_aug[feats].to_numpy(float)
            w = make_sample_weights_by_uniques(X_comb, decimals=args.dedup_decimals, target_total_weight=target_total_weight)
            train_versions.append(("fixunique", df_tr_aug, w))

            # 2) Random ROI baseline（在 raw 上做即可）
            rng = np.random.RandomState(fold_seed + 1000)
            for k in k_list:
                for rep in tqdm(range(args.random_roi_reps), desc="RandomROI", leave=False):
                    rois = rng.choice(feats, size=k, replace=False).tolist()
                    met = eval_metrics(df_tr_aug, df_va, rois, seed=fold_seed + rep)
                    rand_rows.append({"fold": fold_id, "method": method, "k": k, "rep": rep, **met})

            # 3) Oracle 单法（内层CV挑 best method+k）
            best = oracle_select(df_tr_aug, feats, seed=fold_seed + 3000, k_list=oracle_k_list, inner_splits=3)
            oracle_rows.append({
                "fold": fold_id, "method": method,
                "oracle_method": best["method"], "oracle_k": best["k"], "oracle_inner_auprc": best["score"],
                "dup_to_real": dup_to_real, "dup_within": dup_within
            })

            # 4) 稳定性选择 + 外层性能（对每个公平版本都做）
            for fair_tag, df_train_use, w_use in tqdm(train_versions, desc="Fairness", leave=False):
                k_list_use = tuple(sorted(set([k for k in k_list])))
                cache_path = _bootstrap_cache_path(
                    cache_dir, fold_id, method, fair_tag,
                    seed=fold_seed + 4000, B=args.bootstrap_B,
                    k_list=k_list_use, gnn_weight=args.gnn_weight
                )

                if args.cache_bootstrap and cache_path.exists():
                    cached = load_bootstrap_cache(cache_path)
                    topk_sets, freq = cached["topk_sets"], cached["freq"]
                else:
                    topk_sets, freq = bootstrap_stability(
                        df_train_use, feats, seed=fold_seed + 4000,
                        B=args.bootstrap_B, k_list=k_list_use,
                        gnn_weight=args.gnn_weight
                    )
                    if args.cache_bootstrap:
                        save_bootstrap_cache(cache_path, topk_sets, freq)

                # 4.1 稳定性（Jaccard）
                for scheme in ["consensus","fusion","gnn"]:
                    for k in k_list:
                        jm, jsd = pairwise_jaccard_mean_sd(topk_sets[(scheme,k)])
                        stab_rows.append({
                            "fold": fold_id, "method": method, "fair": fair_tag,
                            "scheme": scheme, "k": k,
                            "jaccard_mean": jm, "jaccard_sd": jsd,
                            "dup_to_real": dup_to_real, "dup_within": dup_within,
                            "B": args.bootstrap_B
                        })

                # 4.2 频率累计（Top10）
                for scheme in ["consensus","fusion","gnn"]:
                    key = (method, scheme, fair_tag)
                    freq_acc.setdefault(key, {})
                    freq_total[key] = freq_total.get(key, 0) + args.bootstrap_B
                    for roi, c in freq[scheme].items():
                        freq_acc[key][roi] = freq_acc[key].get(roi, 0) + int(c)

                # 4.3 外层验证fold性能：用“频率排序的稳定Top-k”
                for scheme in ["consensus","fusion","gnn"]:
                    fdict = freq[scheme]
                    stable_rank = sorted(feats, key=lambda x: (fdict.get(x, 0), x), reverse=True)

                    for k in k_list:
                        rois = stable_rank[:k]
                        met = eval_metrics(df_train_use, df_va, rois, seed=fold_seed + 5000, sample_weight=w_use)
                        perf_rows.append({
                            "fold": fold_id, "seed": fold_seed,
                            "method": method, "fair": fair_tag, "scheme": scheme, "k": k,
                            **met,
                            "dup_to_real": dup_to_real, "dup_within": dup_within,
                            "train_n": int(len(df_train_use)),
                            "train_weight_sum": float(np.sum(w_use)) if w_use is not None else float("nan")
                        })

    # ========== 输出 ==========
    df_perf = pd.DataFrame(perf_rows)
    df_perf.to_csv(out_dir / "perf_long.csv", index=False, encoding="utf-8-sig")

    df_perf_sum = (df_perf
                   .groupby(["method","fair","scheme","k"])
                   .agg(
                        auc_mean=("auc","mean"), auc_sd=("auc","std"),
                        auprc_mean=("auprc","mean"), auprc_sd=("auprc","std"),
                        mcc_mean=("mcc","mean"), mcc_sd=("mcc","std"),
                        balacc_mean=("balacc","mean"), balacc_sd=("balacc","std"),
                        logloss_mean=("logloss","mean"), logloss_sd=("logloss","std"),
                        brier_mean=("brier","mean"), brier_sd=("brier","std"),
                        dup_to_real_mean=("dup_to_real","mean"), dup_to_real_sd=("dup_to_real","std"),
                        dup_within_mean=("dup_within","mean"), dup_within_sd=("dup_within","std"),
                   ).reset_index())
    df_perf_sum.to_csv(out_dir / "perf_mean_sd.csv", index=False, encoding="utf-8-sig")

    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(out_dir / "stability_long.csv", index=False, encoding="utf-8-sig")

    df_stab_sum = (df_stab
                   .groupby(["method","fair","scheme","k"])
                   .agg(
                        jaccard_mean=("jaccard_mean","mean"),
                        jaccard_sd=("jaccard_mean","std"),
                   ).reset_index())
    df_stab_sum.to_csv(out_dir / "stability_mean_sd.csv", index=False, encoding="utf-8-sig")

    for (method, scheme, fair), fdict in freq_acc.items():
        total = freq_total[(method, scheme, fair)]
        df_f = pd.DataFrame({
            "BrainRegion": list(fdict.keys()),
            "freq": [fdict[r] for r in fdict.keys()],
            "freq_rate": [fdict[r] / (total + 1e-12) for r in fdict.keys()]
        }).sort_values("freq_rate", ascending=False)
        df_f.to_csv(out_dir / f"freq_top10_{method}_{scheme}_{fair}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(rand_rows).to_csv(out_dir / "random_roi_baseline.csv", index=False, encoding="utf-8-sig")

    df_or = pd.DataFrame(oracle_rows)
    df_or.to_csv(out_dir / "oracle_long.csv", index=False, encoding="utf-8-sig")
    df_or_sum = (df_or.groupby(["method","oracle_method","oracle_k"])
                 .agg(n=("fold","count"), inner_auprc_mean=("oracle_inner_auprc","mean"))
                 .reset_index()
                 .sort_values("inner_auprc_mean", ascending=False))
    df_or_sum.to_csv(out_dir / "oracle_summary.csv", index=False, encoding="utf-8-sig")

    print("\nDone. Results saved to:", out_dir.resolve())
    print("Key files:")
    print("  perf_long.csv / perf_mean_sd.csv")
    print("  stability_long.csv / stability_mean_sd.csv")
    print("  freq_top10_<method>_<scheme>_<fair>.csv")
    print("  random_roi_baseline.csv")
    print("  oracle_long.csv / oracle_summary.csv")


if __name__ == "__main__":
    main()
