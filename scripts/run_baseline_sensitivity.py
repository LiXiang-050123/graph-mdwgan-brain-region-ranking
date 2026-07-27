# -*- coding: utf-8 -*-
r"""
run_baseline_sensitivity.py

功能：
1) 复用旧脚本每fold已训练好的 Ours ckpt（五折记忆）
2) 三个“必做”实验：
   - Permutation test（标签置换负对照）
   - Real-only（不扩增对照）
   - Factor sensitivity（多扩增倍数：如 2/4/8）
3) 下游预测验证：三分类器
   - Logistic Regression
   - Linear SVM (probability=True)
   - RandomForest
4) 公平对照：raw / dedup / fixunique（默认都做）
5) Bootstrap稳定性选择（频率核心集合 + Jaccard）

强约束（与旧脚本对齐以复用记忆）：
- 外层 CV：RepeatedStratifiedKFold(random_state=cv_seed, n_splits, n_repeats)
- fold_seed：cv_seed + fold_id * 17
- 旧ckpt目录：reuse_cv_dir/fold_0001/ckpt_ours/{Saline,METH}/*.pt

运行示例：
python run_baseline_sensitivity.py ^
    --excel data/brain_region_cfos.xlsx ^
  --reuse_cv_dir "cv_results" ^
  --factor_list 2,4,8 ^
  --n_splits 5 --n_repeats 5 --cv_seed 20250829 ^
  --bootstrap_B 50 --out_dir "onezone_final_results"
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

# --------- 全局警告抑制 ---------
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=DataConversionWarning)
except Exception:
    pass

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

from graph_mdwgan import load_excel_data, train_group, EPS_POS

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
    keep = np.array([t not in real_set for t in syn_t], dtype=bool)
    syn = syn.loc[keep].copy()
    syn["_k"] = round_tuples(syn[feats].to_numpy(float), decimals)
    syn = syn.drop_duplicates(subset=["_k"]).drop(columns=["_k"])
    return syn


# ----------------- 复用 ckpt：定位 + 加载 + 采样 -----------------
def _resolve_ckpt_path(ckpt_dir: Path, group: str) -> str:
    group_dir = ckpt_dir / group
    if not group_dir.exists():
        raise FileNotFoundError(f"找不到 {group_dir}")
    candidates = sorted(group_dir.glob("best*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(group_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"{group_dir} 下没有任何 .pt 文件")
    return str(candidates[0])

def _load_generator_from_ckpt(ckpt_path: str, device: str):
    from graph_mdwgan import Generator
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("model_config", {})
    feat_dim = cfg.get("feat_dim")
    if feat_dim is None:
        raise ValueError("ckpt 缺少 feat_dim")

    G = Generator(
        z_dim=cfg.get("z_dim", 128),
        feat_dim=feat_dim,
        hidden=cfg.get("g_hidden"),
        gm_width=cfg.get("gm_width"),
        dropout=cfg.get("dropout", 0.0)
    ).to(device)
    G.load_state_dict(ck["G"])
    G.eval()

    if "ema" in ck and isinstance(ck["ema"], dict):
        with torch.no_grad():
            for n, p in G.named_parameters():
                if p.requires_grad and (n in ck["ema"]):
                    p.data.copy_(torch.tensor(ck["ema"][n], device=device, dtype=p.dtype))

    Ahat = torch.tensor(ck["Ahat"], device=device, dtype=torch.float32)
    mean = torch.tensor(ck["scaler_mean"], device=device, dtype=torch.float32)
    scale = torch.tensor(ck["scaler_scale"], device=device, dtype=torch.float32)
    return G, Ahat, mean, scale

@torch.no_grad()
def _generate_synthetic_before(G, Ahat, mean, scale, n_samples: int, device: str, batch_size: int = 128):
    if hasattr(G, "fc1") and hasattr(G.fc1, "in_features"):
        Z_DIM = G.fc1.in_features
    else:
        Z_DIM = 128
    outs = []
    remain = int(n_samples)
    while remain > 0:
        b = min(batch_size, remain)
        z = torch.randn(b, Z_DIM, device=device)
        fk_std = G(z, Ahat)
        fk_log = fk_std * scale + mean
        fk_before = torch.expm1(fk_log).clamp_min(EPS_POS)
        outs.append(fk_before.detach().cpu().numpy().astype(np.float32))
        remain -= b
    return np.vstack(outs) if outs else np.zeros((0, int(mean.shape[0])), dtype=np.float32)

def make_ours_groupwise_reuse_ckpt(df_tr_real: pd.DataFrame, factor: int, fold_seed: int,
                                  fold_dir_old: Path, strict_retrain: bool):
    """
    优先复用旧fold目录中的 ckpt；缺失则重训并写回同目录（便于下次复用）。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = feat_cols(df_tr_real)

    ckpt_dir = fold_dir_old / "ckpt_ours"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    X_train_all = df_tr_real[feats].to_numpy(float)
    Ahat_np = safe_corrcoef(np.log1p(np.clip(X_train_all, 0, None)))

    dfS = df_tr_real[df_tr_real["Group"] == "Saline"]
    dfM = df_tr_real[df_tr_real["Group"] == "METH"]
    XS = dfS[feats].to_numpy(float)
    XM = dfM[feats].to_numpy(float)

    ckpt_S = ckpt_M = None
    if not strict_retrain:
        try:
            ckpt_S = _resolve_ckpt_path(ckpt_dir, "Saline")
            ckpt_M = _resolve_ckpt_path(ckpt_dir, "METH")
        except Exception:
            ckpt_S = ckpt_M = None

    if ckpt_S is None or ckpt_M is None:
        # 严格无泄漏：只用本fold训练集训练生成器
        train_group("Saline", XS, Ahat_np, seed=fold_seed,   out_dir=str(ckpt_dir), force_retrain=True)
        train_group("METH",   XM, Ahat_np, seed=fold_seed+1, out_dir=str(ckpt_dir), force_retrain=True)
        ckpt_S = _resolve_ckpt_path(ckpt_dir, "Saline")
        ckpt_M = _resolve_ckpt_path(ckpt_dir, "METH")

    GS, AhatS, meanS, scaleS = _load_generator_from_ckpt(ckpt_S, device=device)
    GM, AhatM, meanM, scaleM = _load_generator_from_ckpt(ckpt_M, device=device)

    n_add_S = XS.shape[0] * (factor - 1)
    n_add_M = XM.shape[0] * (factor - 1)

    synS = _generate_synthetic_before(GS, AhatS, meanS, scaleS, n_add_S, device=device)
    synM = _generate_synthetic_before(GM, AhatM, meanM, scaleM, n_add_M, device=device)

    df_synS = pd.DataFrame(synS, columns=feats)
    df_synS["Group"] = "Saline"
    df_synS["Origin"] = "Synthetic"
    df_synS["SampleID"] = [f"Saline_Ours_{i+1:05d}" for i in range(len(df_synS))]

    df_synM = pd.DataFrame(synM, columns=feats)
    df_synM["Group"] = "METH"
    df_synM["Origin"] = "Synthetic"
    df_synM["SampleID"] = [f"METH_Ours_{i+1:05d}" for i in range(len(df_synM))]

    df_tr_aug = pd.concat([df_tr_real, df_synS[df_tr_real.columns], df_synM[df_tr_real.columns]],
                          ignore_index=True)
    df_added = pd.concat([df_synS[df_tr_real.columns], df_synM[df_tr_real.columns]], ignore_index=True)

    # 估算dup（用于写作解释）
    dtr_S, dwi_S = dup_rates(XS, synS)
    dtr_M, dwi_M = dup_rates(XM, synM)
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


# ----------------- 轻量筛选（用于bootstrap） -----------------
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
        n_estimators=400, random_state=seed,
        class_weight="balanced_subsample", n_jobs=-1
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

def consensus_and_fusion(scores_dict: dict, gnn_weight: float):
    methods = list(scores_dict.keys())
    ranks = []
    for m in methods:
        s = scores_dict[m]
        order = np.argsort(-s)
        r = np.empty_like(order)
        r[order] = np.arange(1, len(s) + 1)
        ranks.append(r.astype(float))
    avg_rank = np.mean(np.vstack(ranks), axis=0)
    cons_score = (avg_rank.max() - avg_rank + 1.0)

    gnn_score = scores_dict.get("GNN", None)
    if gnn_score is None:
        return cons_score, cons_score

    def norm01(v):
        v = np.asarray(v, float)
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn + 1e-12)

    fusion = (1 - gnn_weight) * norm01(cons_score) + gnn_weight * norm01(gnn_score)
    return cons_score, fusion

def bootstrap_stability_with_y(train_df, feats, y_train, seed, B, k_list, gnn_weight):
    rng = np.random.RandomState(seed)
    X = train_df[feats].to_numpy(float)
    n = len(train_df)

    topk_sets = {(scheme,k): [] for scheme in ["consensus","fusion","gnn"] for k in k_list}
    freq = {scheme: {} for scheme in ["consensus","fusion","gnn"]}

    for b in tqdm(range(B), desc="Bootstrap", leave=False):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        Xb = X[idx]; yb = y_train[idx]

        scores = {
            "ANOVA": rank_anova(Xb, yb),
            "MI":    rank_mi(Xb, yb, seed=seed + 10 + b),
            "LASSO": rank_lasso(Xb, yb, seed=seed + 20 + b),
            "RF":    rank_rf(Xb, yb, seed=seed + 30 + b),
            "RFE":   rank_rfe_svm(Xb, yb, seed=seed + 40 + b),
            "GNN":   rank_gnn_lite_proxy(Xb, yb, seed=seed + 50 + b),
        }
        cons_score, fus_score = consensus_and_fusion(scores, gnn_weight=gnn_weight)
        gnn_score = scores["GNN"]

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

        kfreq = 10 if 10 in k_list else max(k_list)
        for roi in fus_rank[:kfreq]:
            freq["fusion"][roi] = freq["fusion"].get(roi, 0) + 1
        for roi in cons_rank[:kfreq]:
            freq["consensus"][roi] = freq["consensus"].get(roi, 0) + 1
        for roi in gnn_rank[:kfreq]:
            freq["gnn"][roi] = freq["gnn"].get(roi, 0) + 1

    return topk_sets, freq

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


# ----------------- 下游预测：三分类器 -----------------
def _fit_predict_proba(clf_name: str, Xtr, ytr, Xva, seed: int, sample_weight=None):
    if clf_name == "logreg":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=20000,
            class_weight="balanced", random_state=seed
        )
        clf.fit(Xtr_s, ytr, sample_weight=sample_weight)
        prob = clf.predict_proba(Xva_s)[:, 1]
        return prob

    if clf_name == "svm":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        clf = SVC(
            kernel="linear", probability=True,
            class_weight="balanced", random_state=seed
        )
        clf.fit(Xtr_s, ytr, sample_weight=sample_weight)
        prob = clf.predict_proba(Xva_s)[:, 1]
        return prob

    if clf_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=600, random_state=seed,
            class_weight="balanced_subsample", n_jobs=-1
        )
        clf.fit(Xtr, ytr, sample_weight=sample_weight)
        prob = clf.predict_proba(Xva)[:, 1]
        return prob

    raise ValueError(f"未知分类器：{clf_name}")

def eval_metrics_3clf(train_df, val_df, rois, seed: int, sample_weight=None, y_train_override=None):
    Xtr = train_df[rois].to_numpy(float)
    Xva = val_df[rois].to_numpy(float)
    ytr = y_train_override if y_train_override is not None else y_from_group(train_df["Group"])
    yva = y_from_group(val_df["Group"])

    if len(np.unique(yva)) < 2:
        return {c: {"auc": np.nan, "auprc": np.nan, "mcc": np.nan, "balacc": np.nan, "logloss": np.nan, "brier": np.nan}
                for c in ["logreg","svm","rf"]}

    out = {}
    for clf_name in ["logreg","svm","rf"]:
        prob = _fit_predict_proba(clf_name, Xtr, ytr, Xva, seed=seed, sample_weight=sample_weight)
        pred = (prob >= 0.5).astype(int)

        met = {}
        met["auc"] = float(roc_auc_score(yva, prob))
        met["auprc"] = float(average_precision_score(yva, prob))
        met["mcc"] = float(matthews_corrcoef(yva, pred))
        met["balacc"] = float(balanced_accuracy_score(yva, pred))
        met["logloss"] = float(log_loss(yva, np.vstack([1-prob, prob]).T, labels=[0,1]))
        met["brier"] = float(np.mean((prob - yva) ** 2))
        out[clf_name] = met
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--excel", type=str, default="data/brain_region_cfos.xlsx")
    ap.add_argument("--sheet", type=str, default=None)

    ap.add_argument("--reuse_cv_dir", type=str, default="outputs/cv_results",
                    help="旧脚本输出目录（包含 fold_0001/ckpt_ours/...）用于复用Ours记忆")

    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=20250829)

    ap.add_argument("--factor_list", type=str, default="2,4,8")
    ap.add_argument("--methods", type=str, default="Ours,ROS,SMOTE,RealOnly")
    ap.add_argument("--smote_k", type=int, default=5)

    ap.add_argument("--k_list", type=str, default="1,3,5,10,20")
    ap.add_argument("--bootstrap_B", type=int, default=50)
    ap.add_argument("--gnn_weight", type=float, default=1.0/6.0)

    ap.add_argument("--dedup_decimals", type=int, default=6)
    ap.add_argument("--strict_ours", action="store_true",
                    help="不复用记忆，强制重训Ours生成器（一般不需要）")

    # Permutation：为提速，默认只在 fusion+fixunique 上做，并且只做小k
    ap.add_argument("--do_permutation", action="store_true")
    ap.add_argument("--perm_reps", type=int, default=50)
    ap.add_argument("--perm_k_list", type=str, default="1,3,5,10")
    ap.add_argument("--perm_bootstrap_B", type=int, default=30)

    ap.add_argument("--out_dir", type=str, default="outputs/baseline_sensitivity")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reuse_dir = Path(args.reuse_cv_dir)
    if not reuse_dir.exists():
        raise FileNotFoundError(f"找不到 reuse_cv_dir: {reuse_dir}")

    factor_list = [int(x) for x in args.factor_list.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    perm_k_list = [int(x) for x in args.perm_k_list.split(",") if x.strip()]

    set_seed(args.cv_seed)

    # 读数据（顺序：先Saline后METH，与旧脚本一致）
    Xs, Xm, roi_names = load_excel_data(args.excel, args.sheet)
    rows = []
    for i in range(Xs.shape[0]):
        r: dict[str, object] = {"Group":"Saline","Origin":"RealTrain","SampleID":f"Saline_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            r[str(roi)] = float(Xs[i,j])
        rows.append(r)
    for i in range(Xm.shape[0]):
        r: dict[str, object] = {"Group":"METH","Origin":"RealTrain","SampleID":f"METH_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            r[str(roi)] = float(Xm[i,j])
        rows.append(r)
    df_all = pd.DataFrame(rows)
    feats = feat_cols(df_all)
    y_all = y_from_group(df_all["Group"])

    rskf = RepeatedStratifiedKFold(
        n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.cv_seed
    )

    perf_rows = []
    stab_rows = []
    perm_rows = []

    # 频率累计：按 factor 分开
    freq_acc = {}   # key=(factor,method,scheme,fair)-> dict roi->count
    freq_total = {} # key-> total bootstrap runs

    fold_id = 0
    for tr_idx, va_idx in tqdm(
        rskf.split(df_all[feats].to_numpy(float), y_all),
        total=args.n_splits * args.n_repeats,
        desc="CV Folds"
    ):
        fold_id += 1
        fold_seed = args.cv_seed + fold_id * 17  # 与旧脚本对齐
        set_seed(fold_seed)

        df_tr_real = df_all.iloc[tr_idx].copy()
        df_va = df_all.iloc[va_idx].copy()
        df_tr_real["Origin"] = "RealTrain"

        fold_dir_old = reuse_dir / f"fold_{fold_id:04d}"

        for factor in tqdm(factor_list, desc="Factors", leave=False):
            for method in tqdm(methods, desc="Methods", leave=False):
                mtag = method.upper()

                # ---- 构造扩增训练集（或Real-only）----
                if mtag == "REALONLY":
                    df_tr_aug = df_tr_real.copy()
                    df_added = pd.DataFrame(columns=df_tr_real.columns)
                    dup_to_real, dup_within = 0.0, 0.0
                elif mtag == "OURS":
                    df_tr_aug, df_added, dup_to_real, dup_within = make_ours_groupwise_reuse_ckpt(
                        df_tr_real, factor, fold_seed, fold_dir_old, strict_retrain=args.strict_ours
                    )
                elif mtag == "ROS":
                    df_tr_aug, df_added = make_ros_groupwise(df_tr_real, factor, seed=fold_seed + 3)
                    dup_to_real, dup_within = dup_rates(
                        df_tr_real[feats].to_numpy(float),
                        df_added[feats].to_numpy(float),
                        decimals=args.dedup_decimals
                    )
                elif mtag == "SMOTE":
                    df_tr_aug, df_added = make_smote_groupwise(df_tr_real, factor, seed=fold_seed + 5, k_neighbors=args.smote_k)
                    dup_to_real, dup_within = dup_rates(
                        df_tr_real[feats].to_numpy(float),
                        df_added[feats].to_numpy(float),
                        decimals=args.dedup_decimals
                    )
                else:
                    raise ValueError(f"未知方法: {method}")

                # ---- 公平版本：raw / dedup / fixunique ----
                target_total_weight = float(len(df_tr_real) * factor)
                train_versions: list[tuple[str, pd.DataFrame, np.ndarray | None]] = [("raw", df_tr_aug, None)]

                # dedup（只去掉与真实相同及合成内部重复）
                if len(df_added) > 0:
                    df_syn = df_tr_aug[df_tr_aug["Origin"] != "RealTrain"].copy()
                    df_syn_d = dedup_against_real(df_tr_real, df_syn, decimals=args.dedup_decimals)
                    df_dedup = pd.concat([df_tr_real, df_syn_d[df_tr_real.columns]], ignore_index=True)
                else:
                    df_dedup = df_tr_real.copy()
                train_versions.append(("dedup", df_dedup, None))

                # fixunique（对 raw 赋权）
                w_fix = make_sample_weights_by_uniques(
                    df_tr_aug[feats].to_numpy(float),
                    decimals=args.dedup_decimals,
                    target_total_weight=target_total_weight
                )
                train_versions.append(("fixunique", df_tr_aug, w_fix))

                # ---- 置换标签（Permutation test）：只对 fusion+fixunique 做（提速且足够硬）----
                if args.do_permutation:
                    rngp = np.random.RandomState(fold_seed + 9000 + factor)
                    # 用 fixunique 分支做置换
                    df_perm_train = df_tr_aug
                    w_perm = w_fix
                    y_true_train = y_from_group(df_perm_train["Group"])

                    for pr in tqdm(range(args.perm_reps), desc="Permutation", leave=False):
                        y_perm = y_true_train.copy()
                        rngp.shuffle(y_perm)

                        topk_sets_p, freq_p = bootstrap_stability_with_y(
                            df_perm_train, feats, y_perm,
                            seed=fold_seed + 9100 + pr,
                            B=args.perm_bootstrap_B,
                            k_list=perm_k_list,
                            gnn_weight=args.gnn_weight
                        )
                        # 用 fusion 的频率生成稳定rank
                        fdict = freq_p["fusion"]
                        stable_rank = sorted(feats, key=lambda x: (fdict.get(x, 0), x), reverse=True)

                        for k in perm_k_list:
                            rois = stable_rank[:k]
                            mets = eval_metrics_3clf(
                                df_perm_train, df_va, rois,
                                seed=fold_seed + 9200 + pr,
                                sample_weight=w_perm,
                                y_train_override=y_perm
                            )
                            for clf_name, met in mets.items():
                                perm_rows.append({
                                    "fold": fold_id, "seed": fold_seed, "factor": factor,
                                    "method": method, "fair": "fixunique", "scheme": "fusion",
                                    "k": k, "perm_rep": pr, "clf": clf_name,
                                    **met
                                })

                # ---- 主流程：稳定性选择 + 三分类器预测验证 ----
                for fair_tag, df_train_use, w_use in train_versions:
                    y_train_use = y_from_group(df_train_use["Group"])

                    topk_sets, freq = bootstrap_stability_with_y(
                        df_train_use, feats, y_train_use,
                        seed=fold_seed + 4000 + factor,
                        B=args.bootstrap_B,
                        k_list=k_list,
                        gnn_weight=args.gnn_weight
                    )

                    # 稳定性（Jaccard）
                    for scheme in ["consensus","fusion","gnn"]:
                        for k in k_list:
                            jm, jsd = pairwise_jaccard_mean_sd(topk_sets[(scheme,k)])
                            stab_rows.append({
                                "fold": fold_id, "seed": fold_seed, "factor": factor,
                                "method": method, "fair": fair_tag, "scheme": scheme, "k": k,
                                "jaccard_mean": jm, "jaccard_sd": jsd,
                                "dup_to_real": dup_to_real, "dup_within": dup_within,
                                "B": args.bootstrap_B
                            })

                    # 频率累计（Top10）
                    for scheme in ["consensus","fusion","gnn"]:
                        key = (factor, method, scheme, fair_tag)
                        freq_acc.setdefault(key, {})
                        freq_total[key] = freq_total.get(key, 0) + args.bootstrap_B
                        for roi, c in freq[scheme].items():
                            freq_acc[key][roi] = freq_acc[key].get(roi, 0) + int(c)

                    # 外层验证fold性能：用“频率排序的稳定Top-k”
                    for scheme in ["consensus","fusion","gnn"]:
                        fdict = freq[scheme]
                        stable_rank = sorted(feats, key=lambda x: (fdict.get(x, 0), x), reverse=True)

                        for k in k_list:
                            rois = stable_rank[:k]
                            mets = eval_metrics_3clf(
                                df_train_use, df_va, rois,
                                seed=fold_seed + 5000 + factor,
                                sample_weight=w_use
                            )
                            for clf_name, met in mets.items():
                                perf_rows.append({
                                    "fold": fold_id, "seed": fold_seed, "factor": factor,
                                    "method": method, "fair": fair_tag, "scheme": scheme, "k": k,
                                    "clf": clf_name,
                                    **met,
                                    "dup_to_real": dup_to_real, "dup_within": dup_within,
                                    "train_n": int(len(df_train_use)),
                                    "train_weight_sum": float(np.sum(w_use)) if w_use is not None else float("nan")
                                })

    # ----------------- 输出 -----------------
    df_perf = pd.DataFrame(perf_rows)
    df_perf.to_csv(out_dir / "perf_long.csv", index=False, encoding="utf-8-sig")

    df_perf_sum = (df_perf
                   .groupby(["factor","method","fair","scheme","k","clf"])
                   .agg(
                       auc_mean=("auc","mean"), auc_sd=("auc","std"),
                       auprc_mean=("auprc","mean"), auprc_sd=("auprc","std"),
                       mcc_mean=("mcc","mean"), mcc_sd=("mcc","std"),
                       balacc_mean=("balacc","mean"), balacc_sd=("balacc","std"),
                       logloss_mean=("logloss","mean"), logloss_sd=("logloss","std"),
                       brier_mean=("brier","mean"), brier_sd=("brier","std"),
                       dup_to_real_mean=("dup_to_real","mean"),
                       dup_within_mean=("dup_within","mean"),
                   ).reset_index())
    df_perf_sum.to_csv(out_dir / "perf_mean_sd.csv", index=False, encoding="utf-8-sig")

    df_stab = pd.DataFrame(stab_rows)
    df_stab.to_csv(out_dir / "stability_long.csv", index=False, encoding="utf-8-sig")

    df_stab_sum = (df_stab
                   .groupby(["factor","method","fair","scheme","k"])
                   .agg(
                       jaccard_mean=("jaccard_mean","mean"),
                       jaccard_sd=("jaccard_mean","std"),
                   ).reset_index())
    df_stab_sum.to_csv(out_dir / "stability_mean_sd.csv", index=False, encoding="utf-8-sig")

    # 频率表（Top10）
    for (factor, method, scheme, fair), fdict in freq_acc.items():
        total = freq_total[(factor, method, scheme, fair)]
        df_f = pd.DataFrame({
            "BrainRegion": list(fdict.keys()),
            "freq": [fdict[r] for r in fdict.keys()],
            "freq_rate": [fdict[r] / (total + 1e-12) for r in fdict.keys()]
        }).sort_values("freq_rate", ascending=False)
        df_f.to_csv(out_dir / f"freq_top10_factor{factor}_{method}_{scheme}_{fair}.csv",
                    index=False, encoding="utf-8-sig")

    # permutation
    if len(perm_rows) > 0:
        df_p = pd.DataFrame(perm_rows)
        df_p.to_csv(out_dir / "permutation_long.csv", index=False, encoding="utf-8-sig")
        df_p_sum = (df_p.groupby(["factor","method","fair","scheme","k","clf"])
                    .agg(
                        auc_mean=("auc","mean"), auc_sd=("auc","std"),
                        auprc_mean=("auprc","mean"), auprc_sd=("auprc","std"),
                        mcc_mean=("mcc","mean"), mcc_sd=("mcc","std"),
                        balacc_mean=("balacc","mean"), balacc_sd=("balacc","std"),
                        logloss_mean=("logloss","mean"), logloss_sd=("logloss","std"),
                        brier_mean=("brier","mean"), brier_sd=("brier","std"),
                    ).reset_index())
        df_p_sum.to_csv(out_dir / "permutation_mean_sd.csv", index=False, encoding="utf-8-sig")

    print("\nDone. Outputs in:", out_dir.resolve())
    print("Key files:")
    print("  perf_long.csv / perf_mean_sd.csv")
    print("  stability_mean_sd.csv")
    print("  freq_top10_factor*_*.csv")
    if len(perm_rows) > 0:
        print("  permutation_long.csv / permutation_mean_sd.csv")


if __name__ == "__main__":
    main()
