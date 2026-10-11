#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_selection.py

完整版特征筛选（适配 integrated_split_augment.py 生成的 CSV：train_augmented_x*.csv）：
- 输入：Group/Origin/SampleID + ROI... ；标签用 Group（Saline=0, METH=1）
- 可选择使用 RealTrain / Synthetic / Both（默认 Both）
- 预处理：异常样本剔除（zscore）、标准化
- 五种经典筛选：ANOVA、LASSO(L1)、RandomForest、RFE+LinearSVM、Mutual Information
- 共识排名（名次平均） + 归一化可视化
- GNN-lite：基于训练数据的 ROI 相关性构图（kNN + 对称归一化），图平滑后 Logistic 回归，|coef| 作为重要性
- 融合排名：Consensus + GNN-lite（默认 5/6 : 1/6）
- 置换重要性（交叉验证）：RF 必算，可选 LASSO/LASSO-CV
- SHAP：RandomForest + LASSO（默认展示 “全部 ROI”，并导出每个 ROI 的 mean(|SHAP|) CSV，避免忽略尾部特征）
- 汇总：summary.csv

用法示例：
python feature_selection.py \
  --csv split_aug_outputs/train_augmented_x8.csv \
  --use_origin Both --topk 20 --with_perm_imp --lasso_cv \
  --gnn_k 8 --gnn_weight 0.1667 --max_display all --out feat_select_results

依赖：numpy, pandas, scikit-learn, matplotlib, shap (可选)
"""

# ---- SHAP 与 NumPy 兼容补丁（避免老版本 SHAP 引用 np.bool）----
import numpy as _np
if not hasattr(_np, "bool"): _np.bool = _np.bool_

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance


# ------------------ 基础 I/O ------------------ #
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_from_train_csv(csv_path: Path, use_origin: str = "Both"):
    """
    读取 train_augmented_x*.csv -> (X, y, roi_names)
    - 必需列：Group, Origin；可有 SampleID；其余均视为 ROI 列
    - 标签：METH=1, Saline=0
    """
    df = pd.read_csv(csv_path)
    need = {"Group", "Origin"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} 缺少必要列 {need}")
    roi_cols = [c for c in df.columns if c not in ("Group", "Origin", "SampleID")]
    if not roi_cols:
        raise ValueError("未检测到 ROI 列（除 Group/Origin/SampleID 之外的数值列）")

    if use_origin not in {"RealTrain", "Synthetic", "Both"}:
        raise ValueError("--use_origin 仅支持 RealTrain / Synthetic / Both")

    if use_origin == "Both":
        sub = df[df["Origin"].isin(["RealTrain", "Synthetic"])].copy()
    else:
        sub = df[df["Origin"] == use_origin].copy()

    y = (sub["Group"].astype(str).str.upper() == "METH").astype(int).values
    X = sub[roi_cols].astype(float).values
    return X, y, roi_cols


def zscore_filter(X, zthr=3.0):
    """基于所有特征的标准化 z 值，剔除任何一个特征超过阈值的样本；返回 X_filtered, keep_mask"""
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    keep = (np.abs(Z) <= zthr).all(axis=1)
    return X[keep], keep


# ------------------ 排名工具 ------------------ #
def rankdata_asc(vals):
    """返回升序名次（越小越好），并列取平均名次"""
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
    # 并列处理
    unique_vals, counts = np.unique(vals, return_counts=True)
    for u, c in zip(unique_vals, counts):
        if c > 1:
            idx = np.where(vals == u)[0]
            ranks[idx] = ranks[idx].mean()
    return ranks


# ------------------ 五种特征方法 ------------------ #
def run_feature_methods(X, y, roi_names, random_state=42, rfe_n_features=None, rf_trees=500, lasso_C=1.0,
                        use_lasso_cv=False):
    results = {}
    n_feat = X.shape[1]

    # 1) ANOVA F & p
    F, p = f_classif(X, y)
    results["ANOVA"] = pd.DataFrame({"BrainRegion": roi_names, "F_value": F, "p_value": p})

    # 2) LASSO (L1-LogReg)
    lasso = LogisticRegression(penalty="l1", C=lasso_C, solver="saga",
                               max_iter=10000, random_state=random_state, class_weight="balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso.fit(X, y)
    coef = lasso.coef_.ravel()
    results["LASSO"] = pd.DataFrame({"BrainRegion": roi_names, "|Coefficient|": np.abs(coef), "Coefficient": coef})

    # 2.1) LASSO-CV（可选）
    lasso_cv = None
    if use_lasso_cv:
        Cs = np.logspace(-3, 3, 25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv = LogisticRegressionCV(
                Cs=Cs,
                cv=min(5, max(2, int(np.bincount(y).min()))),
                penalty="l1", solver="saga", scoring="roc_auc",
                class_weight="balanced", max_iter=8000, n_jobs=-1, refit=True, random_state=random_state
            )
            lasso_cv.fit(X, y)
        coef_cv = lasso_cv.coef_.ravel()
        results["LASSO_CV"] = pd.DataFrame({"BrainRegion": roi_names, "|Coefficient|": np.abs(coef_cv),
                                            "Coefficient": coef_cv, "BestC": float(lasso_cv.C_[0])})

    # 3) RandomForest
    rf = RandomForestClassifier(n_estimators=rf_trees, random_state=random_state, class_weight="balanced")
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    results["RandomForest"] = pd.DataFrame({"BrainRegion": roi_names, "Importance": rf_imp})

    # 4) RFE + Linear SVM
    base = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=random_state)
    if rfe_n_features is None:
        rfe_n_features = max(1, int(np.sqrt(n_feat)))
    rfe = RFE(estimator=base, n_features_to_select=rfe_n_features, step=0.1)
    rfe.fit(X, y)
    rfe_rank = rfe.ranking_.astype(float)
    svm = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=random_state).fit(X, y)
    results["RFE_SVM"] = pd.DataFrame({"BrainRegion": roi_names, "RFE_rank": rfe_rank,
                                       "|SVM_weight|": np.abs(svm.coef_.ravel())})

    # 5) Mutual Information
    mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=3)
    results["MutualInfo"] = pd.DataFrame({"BrainRegion": roi_names, "MI": mi})

    return results, rf, lasso, (lasso_cv if use_lasso_cv else None)


# ------------------ 共识与可视化 ------------------ #
def save_method_tables(results: dict, out_dir: Path, topk: int):
    ensure_dir(out_dir)
    # 保存并画 TopK
    for name, df in results.items():
        asc = False
        value_col = None
        if name == "ANOVA":
            value_col = "F_value"; asc = False
        elif name in ("LASSO", "LASSO_CV"):
            value_col = "|Coefficient|"; asc = False
        elif name == "RandomForest":
            value_col = "Importance"; asc = False
        elif name == "RFE_SVM":
            # 用 RFE_rank 画（名次越小越好），做一个转换
            df = df.copy(); df["score"] = -df["RFE_rank"]; value_col = "score"; asc = False
        elif name == "MutualInfo":
            value_col = "MI"; asc = False

        df.sort_values(value_col if value_col else df.columns[1], ascending=False, inplace=True)
        df.to_csv(out_dir / f"{name}_all.csv", index=False, encoding="utf-8-sig")

        # TopK 图
        try:
            plot_top_bar(df, value_col if value_col else df.columns[1],
                         f"{name} Top-{topk}", out_dir / f"plot_{name.lower()}_topk.png", topk, ascending=asc)
        except Exception:
            pass


def plot_top_bar(df, value_col, title, save_path, topk, ascending=False):
    d = df[["BrainRegion", value_col]].dropna().sort_values(value_col, ascending=ascending).tail(topk)
    d = d.sort_values(value_col, ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(d["BrainRegion"], d[value_col])
    plt.title(title); plt.tight_layout()
    plt.savefig(save_path, dpi=220); plt.close()


def consensus_tables(results: dict, out_dir: Path, topk: int):
    """
    将各方法分数转为“名次”，求平均得到 Consensus。
    同时生成归一化 0-1 分的汇总与可视化（不丢尾部特征）。
    """
    ensure_dir(out_dir)

    # 名次（小优）
    def to_rank(df, col, asc=False):
        r = df[["BrainRegion", col]].copy()
        r["rank"] = r[col].rank(ascending=asc, method="min")
        return r.set_index("BrainRegion")["rank"]

    ranks = {}
    ranks["ANOVA"] = to_rank(results["ANOVA"], "F_value", asc=False)
    ranks["LASSO"] = to_rank(results["LASSO"], "|Coefficient|", asc=False)
    ranks["RF"]    = to_rank(results["RandomForest"], "Importance", asc=False)
    ranks["RFE"]   = to_rank(results["RFE_SVM"], "RFE_rank", asc=True)
    ranks["MI"]    = to_rank(results["MutualInfo"], "MI", asc=False)

    regions = list(results["ANOVA"]["BrainRegion"].values)
    df_cons = pd.DataFrame(index=regions)
    for k, s in ranks.items():
        df_cons[k] = s.reindex(regions)
    df_cons["avg_rank"] = df_cons[["ANOVA", "LASSO", "RF", "RFE", "MI"]].mean(axis=1)
    cons_sorted = df_cons.sort_values("avg_rank", ascending=True).reset_index().rename(columns={"index": "BrainRegion"})
    cons_sorted.to_csv(out_dir / "consensus_ranking.csv", index=False, encoding="utf-8-sig")

    # 归一化 0-1 分数（名次→分数）
    def rank_to_score(rank_series):
        n = rank_series.notna().sum()
        if n <= 1:
            return pd.Series(0.0, index=rank_series.index)
        return 1.0 - (rank_series - 1.0) / (n - 1.0)

    df_scores = pd.DataFrame({
        "ANOVA": rank_to_score(df_cons["ANOVA"]),
        "LASSO": rank_to_score(df_cons["LASSO"]),
        "RF":    rank_to_score(df_cons["RF"]),
        "RFE":   rank_to_score(df_cons["RFE"]),
        "MI":    rank_to_score(df_cons["MI"]),
    }, index=regions)
    df_scores["ConsensusScore"] = df_scores.mean(axis=1, skipna=True)
    df_scores.index.name = "BrainRegion"
    cons_plot_dir = out_dir / "consensus_plots"; ensure_dir(cons_plot_dir)
    df_scores.sort_values("ConsensusScore", ascending=False).to_csv(cons_plot_dir / "consensus_scores.csv",
                                                                    float_format="%.6f")

    # TopK 条形 & 各方法散点（仍然只“显示”TopK，但 CSV 覆盖全部 ROI）
    top = df_scores.sort_values("ConsensusScore", ascending=False).head(min(topk, len(df_scores))).iloc[::-1]
    plt.figure(figsize=(8, 6)); plt.barh(top.index, top["ConsensusScore"])
    plt.title(f"Consensus Importance (Top-{len(top)})"); plt.xlabel("Normalized consensus score (0–1)")
    plt.tight_layout(); plt.savefig(cons_plot_dir / "consensus_bar_topk.png", dpi=220); plt.close()

    methods = ["ANOVA", "LASSO", "RF", "RFE", "MI"]
    plt.figure(figsize=(9, 6)); y = np.arange(len(top.index))
    for m in methods:
        plt.plot(top[m], y, "o", label=m)
    plt.yticks(y, top.index); plt.xlabel("Per-method normalized score (0–1)")
    plt.title(f"Per-method Scores for Top-{len(top)} Regions")
    plt.legend(loc="lower right", ncol=1, fontsize=8)
    plt.tight_layout(); plt.savefig(cons_plot_dir / "consensus_methods_topk.png", dpi=220); plt.close()

    return cons_sorted, df_scores  # 排名表 & 0-1 分表（均含全部 ROI）


# ------------------ GNN-lite ------------------ #
def build_norm_adj(X_train: np.ndarray, k: int = 8) -> np.ndarray:
    """特征相关 + kNN 构图；对称归一化 \hat A"""
    C = np.corrcoef(X_train, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    S = np.abs(C)
    A = np.zeros_like(S, dtype=np.float32)
    for i in range(S.shape[0]):
        idx = np.argsort(S[i])[::-1][:k]
        A[i, idx] = 1.0; A[idx, i] = 1.0
    A = A + np.eye(A.shape[0], dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    return D_inv_sqrt @ A @ D_inv_sqrt


def gnn_lite_rank(X: np.ndarray, y: np.ndarray, names, out_dir: Path, k: int = 8, max_iter: int = 6000):
    """在 K 折上平均 |coef| 作为 GNN-lite 重要性；输出完整 CSV + Top10 图"""
    names = list(names)
    counts = np.bincount(y); min_count = int(counts.min())
    n_splits = min(5, max(2, min_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    coef_sum = np.zeros(X.shape[1], dtype=np.float64)
    for tr, _ in skf.split(X, y):
        Ahat = build_norm_adj(X[tr], k=k)
        Xg = X[tr] @ Ahat.T
        clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs")
        clf.fit(Xg, y[tr])
        coef_sum += np.abs(clf.coef_[0])

    coef_avg = coef_sum / n_splits
    coef_norm = (coef_avg - coef_avg.min()) / (np.ptp(coef_avg) + 1e-12)

    df = pd.DataFrame({"BrainRegion": names, "GNNlite_coef": coef_avg, "GNNlite_coef_norm": coef_norm}) \
           .sort_values("GNNlite_coef", ascending=False).reset_index(drop=True)

    ensure_dir(out_dir)
    df.to_csv(out_dir / "gnn_lite_importance.csv", index=False, encoding="utf-8-sig")

    topk = min(10, len(df)); top = df.head(topk).iloc[::-1]
    plt.figure(figsize=(8, 6)); plt.barh(top["BrainRegion"], top["GNNlite_coef_norm"])
    plt.title(f"GNN-lite Importance (Top-{topk})"); plt.xlabel("Normalized |coef|")
    plt.tight_layout(); plt.savefig(out_dir / "gnn_lite_importance_top10.png", dpi=220); plt.close()

    return df


def fuse_consensus_gnn(cons_scores_csv: Path, gnn_df: pd.DataFrame, out_dir: Path, gnn_weight: float = 1.0/6.0):
    """融合排名：ConsensusScore 与 GNN-lite 归一分线性加权"""
    cons = pd.read_csv(cons_scores_csv)
    if "BrainRegion" not in cons.columns:
        cons = pd.read_csv(cons_scores_csv, index_col=0).reset_index().rename(columns={"index": "BrainRegion"})
    merged = cons.merge(gnn_df[["BrainRegion", "GNNlite_coef_norm"]], on="BrainRegion", how="left")
    alpha = float(gnn_weight); alpha = max(0.0, min(1.0, alpha))
    merged["ConsensusPlusGNN"] = (1.0 - alpha) * merged["ConsensusScore"] + alpha * merged["GNNlite_coef_norm"]
    merged = merged.sort_values("ConsensusPlusGNN", ascending=False)
    ensure_dir(out_dir)
    merged.to_csv(out_dir / "consensus_plus_gnn.csv", index=False, encoding="utf-8-sig")

    topm = merged.head(min(10, len(merged))).iloc[::-1]
    plt.figure(figsize=(8, 6)); plt.barh(topm["BrainRegion"], topm["ConsensusPlusGNN"])
    plt.title("Consensus + GNN-lite (Top-10)"); plt.xlabel("Fused normalized score")
    plt.tight_layout(); plt.savefig(out_dir / "consensus_plus_gnn_top10.png", dpi=220); plt.close()

    return merged


# ------------------ 置换重要性（CV） ------------------ #
def permutation_importance_cv(estimator_ctor, X: np.ndarray, y: np.ndarray,
                              scoring: str = "roc_auc", n_splits: int = 5,
                              n_repeats: int = 30, random_state: int = 42):
    counts = np.bincount(y); min_count = int(counts.min())
    n_splits = min(n_splits, max(2, min_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    imps = []
    for tr, va in skf.split(X, y):
        model = estimator_ctor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[tr], y[tr])
        pi = permutation_importance(model, X[va], y[va], scoring=scoring, n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=-1)
        imps.append(pi.importances_mean)
    imps = np.stack(imps, axis=0)
    return imps.mean(axis=0), imps.std(axis=0)


# ------------------ 主流程 ------------------ #
def run_pipeline(csv_path, use_origin="Both", out_dir="feat_select_results",
                 topk=20, zthr=3.0, lasso_C=1.0, rf_trees=500,
                 with_perm_imp=False, perm_metric="roc_auc", perm_repeats=50,
                 with_lasso_cv=False, gnn_k=8, gnn_weight=1.0/6.0,
                 max_display="all", seed=42,
                 export_topn="1,3", export_prefix="train_augmented_top"):

    out_dir = Path(out_dir); ensure_dir(out_dir)

    # 1) 载入 & 标签
    X, y, roi_names = load_from_train_csv(Path(csv_path), use_origin=use_origin)

    # 2) 异常样本剔除（可选） + 标准化
    if zthr and zthr > 0:
        X, keep = zscore_filter(X, zthr=zthr)
        pd.DataFrame({"kept": keep}).to_csv(out_dir / "zfilter_mask.csv", index=False)
        y = y[keep]

    scaler = StandardScaler(); Xs = scaler.fit_transform(X)

    # 3) 各方法打分
    results, rf_model, lasso_model, lasso_cv_model = run_feature_methods(
        Xs, y, roi_names, random_state=seed, rf_trees=rf_trees, lasso_C=lasso_C, use_lasso_cv=with_lasso_cv
    )
    methods_dir = out_dir / "methods"; save_method_tables(results, methods_dir, topk)

    # 4) 共识排名 + 归一化图
    cons_rank, cons_scores = consensus_tables(results, out_dir, topk)

    # 5) GNN-lite + 融合
    gnn_dir = out_dir / "gnn_results"
    gnn_df = gnn_lite_rank(Xs, y, roi_names, gnn_dir, k=gnn_k, max_iter=6000)
    fused = fuse_consensus_gnn(out_dir / "consensus_plots/consensus_scores.csv", gnn_df, gnn_dir, gnn_weight=gnn_weight)

    # 5.5) 根据融合排名导出裁剪后的训练集（Top-N 列表，如 1,3）
    try:
        import pandas as _pd
        fused_list = _pd.read_csv(gnn_dir / "consensus_plus_gnn.csv")
        roi_order = fused_list["BrainRegion"].tolist()
        # 安全处理 N 列表
        Ns = [int(x) for x in str(export_topn).split(',') if str(x).strip().isdigit()]
        Ns = sorted(set([n for n in Ns if n >= 1]))
        src_df = _pd.read_csv(csv_path)
        base_cols = [c for c in ["Group","Origin","SampleID"] if c in src_df.columns]
        for N in Ns:
            take = min(N, len(roi_order))
            sel_rois = roi_order[:take]
            cols = base_cols + [c for c in sel_rois if c in src_df.columns]
            out_csv = out_dir / f"{export_prefix}{take}.csv"
            src_df[cols].to_csv(out_csv, index=False, encoding="utf-8-sig")
    except Exception as e:
        (out_dir / "export_topn_error.txt").write_text(str(e), encoding="utf-8")

    # 6) 置换重要性（RF 必算；LASSO/LASSO-CV 可选）
    perm_dir = out_dir / "permutation_importance"
    if with_perm_imp:
        ensure_dir(perm_dir)
        def rf_ctor():
            return RandomForestClassifier(n_estimators=rf_trees, random_state=seed, class_weight="balanced")
        mean_imp, std_imp = permutation_importance_cv(rf_ctor, Xs, y, scoring=perm_metric,
                                                      n_splits=5, n_repeats=perm_repeats, random_state=seed)
        df_rf = pd.DataFrame({"BrainRegion": roi_names, "PermImportance_mean": mean_imp, "PermImportance_std": std_imp}) \
                    .sort_values("PermImportance_mean", ascending=False).reset_index(drop=True)
        df_rf.to_csv(perm_dir / "permutation_importance_rf.csv", index=False, encoding="utf-8-sig")
        plot_top_bar(df_rf, "PermImportance_mean", f"Permutation Importance (RF) Top-{topk}",
                     perm_dir / "plot_permutation_importance_rf_topk.png", topk)

        # LASSO（如需也计算 PI，次数默认较少以控时）
        try:
            def lasso_ctor():
                return LogisticRegression(penalty="l1", C=lasso_C, solver="saga",
                                          max_iter=6000, random_state=seed, class_weight="balanced")
            mean_imp_l, std_imp_l = permutation_importance_cv(lasso_ctor, Xs, y, scoring=perm_metric,
                                                              n_splits=5, n_repeats=max(10, perm_repeats//2), random_state=seed+1)
            df_lasso = pd.DataFrame({"BrainRegion": roi_names, "PermImportance_mean": mean_imp_l, "PermImportance_std": std_imp_l}) \
                        .sort_values("PermImportance_mean", ascending=False).reset_index(drop=True)
            df_lasso.to_csv(perm_dir / "permutation_importance_lasso.csv", index=False, encoding="utf-8-sig")
            plot_top_bar(df_lasso, "PermImportance_mean", f"Permutation Importance (LASSO) Top-{topk}",
                         perm_dir / "plot_permutation_importance_lasso_topk.png", topk)
        except Exception as e:
            (perm_dir / "permutation_importance_lasso_error.txt").write_text(str(e), encoding="utf-8")

        # LASSO-CV 固定最佳 C 后的 PI（如有）
        if with_lasso_cv and (lasso_cv_model is not None):
            try:
                best_C = float(lasso_cv_model.C_[0]) if hasattr(lasso_cv_model, "C_") else lasso_C
                def lasso_cv_ctor():
                    return LogisticRegression(penalty="l1", C=best_C, solver="saga",
                                              max_iter=8000, random_state=seed+2, class_weight="balanced")
                mean_imp_lcv, std_imp_lcv = permutation_importance_cv(lasso_cv_ctor, Xs, y, scoring=perm_metric,
                                                                      n_splits=5, n_repeats=max(10, perm_repeats//2), random_state=seed+2)
                df_lcv = pd.DataFrame({"BrainRegion": roi_names, "PermImportance_mean": mean_imp_lcv, "PermImportance_std": std_imp_lcv}) \
                            .sort_values("PermImportance_mean", ascending=False).reset_index(drop=True)
                df_lcv.to_csv(perm_dir / "permutation_importance_lasso_cv.csv", index=False, encoding="utf-8-sig")
                plot_top_bar(df_lcv, "PermImportance_mean", f"Permutation Importance (LASSO-CV) Top-{topk}",
                             perm_dir / "plot_permutation_importance_lasso_cv_topk.png", topk)
            except Exception as e:
                (perm_dir / "permutation_importance_lasso_cv_error.txt").write_text(str(e), encoding="utf-8")

    # 7) SHAP：展示全部 ROI，并导出 mean(|SHAP|) CSV（避免忽略尾部特征）
    shap_dir = out_dir / "shap"; ensure_dir(shap_dir)
    try:
        import shap
        # max_display 解析
        if isinstance(max_display, str) and max_display.lower() == "all":
            md = len(roi_names)
        else:
            md = int(max_display)

        # RandomForest SHAP
        try:
            expl_rf = shap.TreeExplainer(rf_model)
            sv_rf_all = expl_rf.shap_values(Xs)
            # 分类任务时，可能返回 list；选择正类索引（若二分类）
            if isinstance(sv_rf_all, list):
                sv_rf = sv_rf_all[-1]
            else:
                sv_rf = sv_rf_all
            # 图（全 ROI）
            shap.summary_plot(sv_rf, Xs, feature_names=roi_names, show=False, max_display=md)
            plt.tight_layout(); plt.gcf().savefig(shap_dir / "shap_beeswarm_randomforest.png", dpi=220, bbox_inches="tight"); plt.close()
            shap.summary_plot(sv_rf, Xs, feature_names=roi_names, show=False, max_display=md, plot_type="bar")
            plt.tight_layout(); plt.gcf().savefig(shap_dir / "shap_bar_randomforest.png", dpi=220, bbox_inches="tight"); plt.close()
            # 导出 mean(|SHAP|) 重要性（覆盖全部 ROI）
            shap_mean_rf = np.mean(np.abs(sv_rf), axis=0)
            pd.DataFrame({"BrainRegion": roi_names, "mean_abs_SHAP": shap_mean_rf}) \
                .sort_values("mean_abs_SHAP", ascending=False).to_csv(shap_dir / "shap_mean_abs_randomforest.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            (shap_dir / "shap_randomforest_error.txt").write_text(str(e), encoding="utf-8")

        # LASSO SHAP（线性模型）
        try:
            expl_lr = shap.LinearExplainer(lasso_model, Xs)
            sv_lr = expl_lr.shap_values(Xs)
            shap.summary_plot(sv_lr, Xs, feature_names=roi_names, show=False, max_display=md)
            plt.tight_layout(); plt.gcf().savefig(shap_dir / "shap_beeswarm_lasso.png", dpi=220, bbox_inches="tight"); plt.close()
            shap.summary_plot(sv_lr, Xs, feature_names=roi_names, show=False, max_display=md, plot_type="bar")
            plt.tight_layout(); plt.gcf().savefig(shap_dir / "shap_bar_lasso.png", dpi=220, bbox_inches="tight"); plt.close()
            # 导出 mean(|SHAP|)（覆盖全部 ROI）
            shap_mean_lr = np.mean(np.abs(sv_lr), axis=0)
            pd.DataFrame({"BrainRegion": roi_names, "mean_abs_SHAP": shap_mean_lr}) \
                .sort_values("mean_abs_SHAP", ascending=False).to_csv(shap_dir / "shap_mean_abs_lasso.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            (shap_dir / "shap_lasso_error.txt").write_text(str(e), encoding="utf-8")

    except Exception as e:
        (shap_dir / "shap_import_error.txt").write_text(str(e), encoding="utf-8")

    # 8) 汇总
    pd.DataFrame({
        "Metric": ["Samples_used", "Features", "Use_origin", "z_thr", "rf_trees", "lasso_C", "with_lasso_cv",
                   "gnn_k", "gnn_weight", "perm_metric", "perm_repeats"],
        "Value":  [Xs.shape[0], Xs.shape[1], use_origin, zthr, rf_trees, lasso_C, with_lasso_cv,
                   gnn_k, gnn_weight, perm_metric, perm_repeats]
    }).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    print("✅ 特征筛选完成。主要输出：")
    print(f" - {methods_dir} （各方法完整排名）")
    print(f" - {out_dir/'consensus_ranking.csv'} / {out_dir/'consensus_plots/consensus_scores.csv'}（共识排名/0-1分）")
    print(f" - {gnn_dir/'gnn_lite_importance.csv'} 与 {gnn_dir/'consensus_plus_gnn.csv'}（GNN 及融合）")
    if with_perm_imp:
        print(f" - {perm_dir}（置换重要性）")
    print(f" - {shap_dir}（SHAP 全 ROI 图与 mean(|SHAP|) CSV）")


# ------------------ CLI ------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full feature selection on CSV (train_augmented_x*.csv) with GNN-lite & SHAP (all ROIs).")
    ap.add_argument("--csv", default="", help="路径：train_augmented_x{factor}.csv（如不提供则自动查找）")
    ap.add_argument("--use_origin", default="Both", choices=["RealTrain", "Synthetic", "Both"], help="参与筛选的数据来源")
    ap.add_argument("--out", default="feat_select_results", help="输出目录")
    ap.add_argument("--topk", type=int, default=20, help="可视化条形图的 Top-K（CSV 始终包含全部 ROI）")
    ap.add_argument("--zthr", type=float, default=3.0, help="异常样本剔除阈值（|z|>阈值 将删除样本；<=0 关闭）")
    ap.add_argument("--lasso_C", type=float, default=1.0, help="LASSO(L1) 的 C（越小越强稀疏）")
    ap.add_argument("--rf_trees", type=int, default=500, help="随机森林树数")
    ap.add_argument("--with_perm_imp", action="store_true", help="计算置换重要性（RF + LASSO + 可选 LASSO-CV）")
    ap.add_argument("--perm_metric", default="roc_auc", choices=["roc_auc", "accuracy"], help="置换重要性的评分指标")
    ap.add_argument("--perm_repeats", type=int, default=50, help="每个特征的置换重复次数")
    ap.add_argument("--with_lasso_cv", action="store_true", help="使用 LASSO-CV 自动选 C")
    ap.add_argument("--gnn_k", type=int, default=8, help="GNN-lite：kNN 图的 k")
    ap.add_argument("--gnn_weight", type=float, default=1.0/6.0, help="融合权重：Consensus*(1-w) + GNN*w")
    ap.add_argument("--max_display", default="all", help="SHAP summary 图展示的 ROI 数量，'all' 表示全部")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export_topn", default="1,3", help="导出裁剪后的训练集，逗号分隔的N列表（例如 1,3）")
    ap.add_argument("--export_prefix", default="train_augmented_top", help="导出数据文件名前缀")
    args = ap.parse_args()

    # 自动查找CSV文件
    csv_path = None
    if args.csv and Path(args.csv).exists():
        csv_path = args.csv
    elif not args.csv:
        # 自动搜索默认位置
        default_paths = [
            "split_aug_outputs/train_augmented_x8.csv",
            "./split_aug_outputs/train_augmented_x8.csv", 
            "train_augmented_x8.csv",
            "C:/Users/高子祥/Desktop/机器学习/split_aug_outputs/train_augmented_x8.csv"
        ]
        for p in default_paths:
            if Path(p).exists():
                csv_path = p
                print(f"自动找到文件：{Path(p).resolve()}")
                break
    else:
        csv_path = args.csv

    if not csv_path or not Path(csv_path).exists():
        raise SystemExit(f"找不到CSV文件。请提供 --csv 参数或确保以下位置存在文件：\n" + 
                        "\n".join([f"  - {p}" for p in [
                            "split_aug_outputs/train_augmented_x8.csv",
                            "train_augmented_x8.csv", 
                            "C:/Users/高子祥/Desktop/机器学习/split_aug_outputs/train_augmented_x8.csv"
                        ]]))

    run_pipeline(csv_path=csv_path, use_origin=args.use_origin, out_dir=args.out,
                 topk=args.topk, zthr=args.zthr, lasso_C=args.lasso_C, rf_trees=args.rf_trees,
                 with_perm_imp=args.with_perm_imp, perm_metric=args.perm_metric, perm_repeats=args.perm_repeats,
                 with_lasso_cv=args.with_lasso_cv, gnn_k=args.gnn_k, gnn_weight=args.gnn_weight,
                 max_display=args.max_display, seed=args.seed,
                 export_topn=args.export_topn, export_prefix=args.export_prefix)
