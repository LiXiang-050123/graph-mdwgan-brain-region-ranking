# -*- coding: utf-8 -*-
"""
run_low_data_stress.py

Low-data stress test (learning-curve/downsampling analysis) based on the fold-level
training files produced by the 25-fold cross-validation workflow. The script reuses
outputs/cv_results/fold_****/fs_{METHOD}/cv_top{k}.csv and refits only the downstream
classifiers; generators and feature-ranking procedures are not rerun.

Objective:
Evaluate whether ranking-based models retain performance as the number of real training
samples is progressively reduced.

Design (for each fold, method, k value, and classifier):
- Recover real training samples (Origin==RealTrain) and synthetic samples (Origin==Synthetic)
  from cv_top{k}.csv.
- Construct training sets at real-sample fractions of 0.2/0.4/0.6/0.8/1.0 by default:
  1) real: stratified downsampling of real training samples, retaining at least min_per_class.
  2) aug: add class-matched synthetic samples to the downsampled real set, targeting
          approximately (factor-1) synthetic samples per real sample (default factor=8).
          Sampling with replacement is used when the synthetic pool is insufficient; if a
          class-specific synthetic pool is unavailable, real samples are resampled as a
          documented fallback.

Outputs:
./oneclick_lowdata_results_fast/
- perf_lowdata_long.csv
- perf_lowdata_mean_sd.csv
- run.log
- settings.json

Example:
cd .
python run_low_data_stress.py --do_quick 1

Set --do_quick 0 or provide custom parameters for a more extensive analysis.

Notes:
- The generator is not invoked by this script.
- Non-finite values are imputed using the training-column median before model fitting.
- Combinations with a single-class training subset are skipped and recorded as NaN.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Suppress the scikit-learn penalty deprecation warning without changing model behavior
warnings.filterwarnings(
    "ignore",
    message=r".*'penalty' was deprecated in version 1\.8.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
# Broader fallback filter for environment-specific warning module names
warnings.filterwarnings(
    "ignore",
    message=r".*penalty.*deprecated.*",
    category=FutureWarning,
)


META_COLS = {"Group", "Origin", "SampleID"}


# ------------------------- Logging -------------------------
def init_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lowdata_fast")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ------------------------- Helpers -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def y_from_group(series: pd.Series) -> np.ndarray:
    return (series.astype(str).str.upper().str.strip() == "METH").astype(int).to_numpy()


def feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def sanitize_matrix_train_val(Xtr: np.ndarray, Xva: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = Xtr.astype(np.float64, copy=True)
    Xva = Xva.astype(np.float64, copy=True)

    Xtr[~np.isfinite(Xtr)] = np.nan
    Xva[~np.isfinite(Xva)] = np.nan

    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    i, j = np.where(np.isnan(Xtr))
    if i.size:
        Xtr[i, j] = med[j]
    i, j = np.where(np.isnan(Xva))
    if i.size:
        Xva[i, j] = med[j]

    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)
    return Xtr, Xva


def fit_predict_proba(clf_name: str, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, seed: int,
                      rf_trees: int = 600) -> np.ndarray:
    if clf_name == "logreg":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            max_iter=20000, class_weight="balanced",
            random_state=seed
        )
        clf.fit(Xtr_s, ytr)
        return clf.predict_proba(Xva_s)[:, 1]

    if clf_name == "svm":
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        clf = SVC(
            kernel="linear", probability=True,
            class_weight="balanced",
            random_state=seed
        )
        clf.fit(Xtr_s, ytr)
        return clf.predict_proba(Xva_s)[:, 1]

    if clf_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=int(rf_trees),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva)[:, 1]

    raise ValueError(f"Unknown clf: {clf_name}")


def eval_metrics(train_df: pd.DataFrame, val_df: pd.DataFrame, rois: List[str], clf_name: str,
                 seed: int, rf_trees: int = 600) -> Dict[str, float]:
    Xtr = train_df[rois].to_numpy(float)
    ytr = y_from_group(train_df["Group"])
    Xva = val_df[rois].to_numpy(float)
    yva = y_from_group(val_df["Group"])

    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        return {"auc": float("nan"), "auprc": float("nan"), "bacc": float("nan"), "mcc": float("nan"), "logloss": float("nan")}

    Xtr, Xva = sanitize_matrix_train_val(Xtr, Xva)
    prob = fit_predict_proba(clf_name, Xtr, ytr, Xva, seed=seed, rf_trees=rf_trees)
    prob = np.clip(np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    auc = float(roc_auc_score(yva, prob))
    auprc = float(average_precision_score(yva, prob))
    pred = (prob >= 0.5).astype(int)
    bacc = float(balanced_accuracy_score(yva, pred))
    mcc = float(matthews_corrcoef(yva, pred))
    ll = float(log_loss(yva, np.column_stack([1 - prob, prob]), labels=[0, 1]))
    return {"auc": auc, "auprc": auprc, "bacc": bacc, "mcc": mcc, "logloss": ll}


def autodetect_excel(base_dir: Path) -> Optional[str]:
    xs = sorted(base_dir.glob("*.xlsx"))
    if not xs:
        return None
    for kw in ["脑区", "填充", "ROI", "roi"]:
        for p in xs:
            if kw in p.name:
                return str(p)
    return str(xs[0])


def build_df_all(excel_path: str, sheet: Optional[str], project_dir: Path) -> pd.DataFrame:
    sys.path.insert(0, str(project_dir))
    from graph_mdwgan import load_excel_data  # type: ignore

    Xs, Xm, roi_names = load_excel_data(excel_path, sheet)

    rows: List[Dict[str, object]] = []
    for i in range(Xs.shape[0]):
        r: Dict[str, object] = {"Group":"Saline", "Origin":"Real", "SampleID": f"Saline_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            r[str(roi)] = float(Xs[i, j])
        rows.append(r)
    for i in range(Xm.shape[0]):
        r = {"Group":"METH", "Origin":"Real", "SampleID": f"METH_{i+1:03d}"}
        for j, roi in enumerate(roi_names):
            r[str(roi)] = float(Xm[i, j])
        rows.append(r)
    return pd.DataFrame(rows)


def load_cv_topk(fs_dir: Path, k: int) -> pd.DataFrame:
    fp = fs_dir / f"cv_top{k}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}")
    df = pd.read_csv(fp)
    for c in ["Group", "Origin", "SampleID"]:
        if c not in df.columns:
            raise ValueError(f"{fp} missing column: {c}")
    return df


def stratified_downsample(df_real: pd.DataFrame, frac: float, seed: int, min_per_class: int = 2) -> pd.DataFrame:
    """
    分层下采样，保证每类至少 min_per_class（若该类本身不足则取全部）
    """
    rng = np.random.RandomState(seed)
    parts = []
    for g, sub in df_real.groupby("Group"):
        n = len(sub)
        if n <= 0:
            continue
        target = int(round(n * frac))
        target = max(min_per_class, target)
        target = min(n, target)
        # sample 不放回
        parts.append(sub.sample(n=target, replace=False, random_state=int(rng.randint(0, 10**9))))
    out = pd.concat(parts, ignore_index=True) if parts else df_real.iloc[0:0].copy()
    return out


def build_aug_from_pools(real_sub: pd.DataFrame, syn_pool: pd.DataFrame, factor: int,
                         seed: int) -> pd.DataFrame:
    """
    给定下采样后的真实集 real_sub，以及该fold记忆文件中的合成池 syn_pool，
    目标为每类合成样本数 ≈ (factor-1) * 该类真实数。
    """
    rng = np.random.RandomState(seed)
    out = [real_sub.copy()]
    for g, real_g in real_sub.groupby("Group"):
        n_real = len(real_g)
        n_syn_target = max(0, int((factor - 1) * n_real))
        if n_syn_target == 0:
            continue

        pool_g = syn_pool[syn_pool["Group"].astype(str) == str(g)]
        if len(pool_g) > 0:
            replace = len(pool_g) < n_syn_target
            picked = pool_g.sample(n=n_syn_target, replace=replace, random_state=int(rng.randint(0, 10**9))).copy()
        else:
            # Documented fallback: resample real observations when the class-specific synthetic pool is unavailable
            replace = True
            picked = real_g.sample(n=n_syn_target, replace=replace, random_state=int(rng.randint(0, 10**9))).copy()
            picked["Origin"] = "Synthetic"
            picked["SampleID"] = [f"{g}_FALLBACK_{i+1:05d}" for i in range(len(picked))]

        out.append(picked)

    df_aug = pd.concat(out, ignore_index=True)
    return df_aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default=".")
    ap.add_argument("--reuse_cv_dir", type=str, default="outputs/cv_results")
    ap.add_argument("--excel", type=str, default="data/brain_region_cfos.xlsx")
    ap.add_argument("--sheet", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="outputs/low_data")

    ap.add_argument("--methods", type=str, default="Ours,ROS,SMOTE")
    ap.add_argument("--k_list", type=str, default="10")
    ap.add_argument("--clfs", type=str, default="logreg,svm,rf")

    ap.add_argument("--factor", type=int, default=8)
    ap.add_argument("--min_per_class", type=int, default=2)

    ap.add_argument("--fracs", type=str, default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--rf_trees", type=int, default=600)

    ap.add_argument("--do_quick", type=int, default=1,
                    help="1=use the predefined reduced-compute settings; 0=use the supplied parameter values.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        base_dir = Path.cwd()

    reuse_cv_dir = Path(args.reuse_cv_dir)
    if not reuse_cv_dir.is_absolute():
        reuse_cv_dir = base_dir / reuse_cv_dir
    if not reuse_cv_dir.exists():
        cand = base_dir / "cv_results"
        if cand.exists():
            reuse_cv_dir = cand
        else:
            raise FileNotFoundError(f"Cannot find reuse_cv_dir: {reuse_cv_dir}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir

    logger = init_logger(out_dir)
    logger.info("===== Low-data stress test (FAST, reuse cv_topK memory) =====")
    logger.info(f"base_dir={base_dir}")
    logger.info(f"reuse_cv_dir={reuse_cv_dir}")

    excel_path = args.excel.strip() or (autodetect_excel(base_dir) or "")
    if not excel_path:
        raise FileNotFoundError(f"No excel found under {base_dir}, please pass --excel")
    p_excel = Path(excel_path)
    if not p_excel.exists():
        cand = base_dir / excel_path
        if cand.exists():
            excel_path = str(cand)
        else:
            raise FileNotFoundError(f"Excel not found: {excel_path}")

    # quick preset
    rf_trees = int(args.rf_trees)
    fracs = [float(x) for x in args.fracs.split(",") if x.strip()]
    if args.do_quick == 1:
        rf_trees = 300
        fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

    (out_dir / "settings.json").write_text(json.dumps({
        **vars(args),
        "excel_path_resolved": excel_path,
        "reuse_cv_dir_resolved": str(reuse_cv_dir),
        "rf_trees_effective": rf_trees,
        "fracs_effective": fracs
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    project_dir = base_dir
    set_seed(20250829)

    t0 = time.time()
    df_all = build_df_all(excel_path, args.sheet, project_dir)
    all_ids = set(df_all["SampleID"].astype(str).tolist())
    logger.info(f"Loaded data: n={len(df_all)}, time={time.time()-t0:.2f}s")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    k_list = sorted({int(x) for x in args.k_list.split(",") if str(x).strip().isdigit()})
    clfs = [c.strip() for c in args.clfs.split(",") if c.strip()]

    folds = sorted([d for d in reuse_cv_dir.glob("fold_*") if d.is_dir()])
    if not folds:
        raise FileNotFoundError(f"No fold_* under {reuse_cv_dir}")

    perf_rows = []

    # 仅用第一个 k 恢复 train split（与之前 FAST 脚本一致）
    k0 = k_list[0]

    for fold_dir in folds:
        fold_id = int(fold_dir.name.split("_")[-1])
        fold_seed = 20250829 + fold_id * 17
        set_seed(fold_seed)
        rng = np.random.RandomState(fold_seed)

        # recover train_real_ids
        train_real_ids = None
        for pref in ["OURS", "ROS", "SMOTE"]:
            fs_try = fold_dir / f"fs_{pref}"
            if fs_try.exists():
                try:
                    df_tmp = load_cv_topk(fs_try, k0)
                    ids = df_tmp.loc[df_tmp["Origin"].astype(str).str.contains("Real", case=False), "SampleID"].astype(str).tolist()
                    if ids:
                        train_real_ids = set(ids)
                        break
                except Exception:
                    continue
        if not train_real_ids:
            raise RuntimeError(f"Cannot recover train split from {fold_dir} (missing cv_top{k0}.csv)")

        val_ids = list(sorted(all_ids - set(train_real_ids)))
        train_ids = list(sorted(train_real_ids))

        df_tr_real_full = df_all[df_all["SampleID"].astype(str).isin(train_ids)].copy()
        df_va_full = df_all[df_all["SampleID"].astype(str).isin(val_ids)].copy()

        for method in methods:
            tag = method.strip().upper()
            fs_dir = fold_dir / f"fs_{tag}"
            if not fs_dir.exists():
                raise FileNotFoundError(f"Missing {fs_dir}")

            for k in k_list:
                df_train_mem = load_cv_topk(fs_dir, k)
                rois = [c for c in df_train_mem.columns if c not in META_COLS]

                # 只保留真实/合成两类（兼容大小写）
                is_real = df_train_mem["Origin"].astype(str).str.contains("Real", case=False)
                is_syn = df_train_mem["Origin"].astype(str).str.contains("Synth", case=False)

                df_real_pool = df_train_mem[is_real].copy()
                df_syn_pool = df_train_mem[is_syn].copy()

                # val（真实）
                df_va = df_va_full[["Group", "Origin", "SampleID"] + rois].copy()

                for frac in fracs:
                    # 真实下采样
                    df_real_sub = stratified_downsample(df_real_pool, frac=frac, seed=int(rng.randint(0, 10**9)),
                                                        min_per_class=int(args.min_per_class))
                    df_real_sub = df_real_sub[["Group", "Origin", "SampleID"] + rois]

                    # aug：按 factor 从合成池补齐（不够则有放回抽样）
                    df_aug_sub = build_aug_from_pools(
                        real_sub=df_real_sub,
                        syn_pool=df_syn_pool[["Group", "Origin", "SampleID"] + rois] if len(df_syn_pool) else df_syn_pool,
                        factor=int(args.factor),
                        seed=int(rng.randint(0, 10**9))
                    )
                    df_aug_sub = df_aug_sub[["Group", "Origin", "SampleID"] + rois]

                    # 两种训练模式各跑一次
                    for train_mode, train_df in [("real", df_real_sub), ("aug", df_aug_sub)]:
                        for clf in clfs:
                            met = eval_metrics(train_df, df_va, rois, clf_name=clf, seed=fold_seed, rf_trees=rf_trees)
                            perf_rows.append({
                                "fold": fold_id,
                                "seed": fold_seed,
                                "method": method,
                                "k": int(len(rois)),
                                "frac_real": float(frac),
                                "train_mode": train_mode,
                                "clf": clf,
                                "n_train_real": int((train_df["Origin"].astype(str).str.contains("Real", case=False)).sum()),
                                "n_train_total": int(len(train_df)),
                                "n_val": int(len(df_va)),
                                **met
                            })

        if fold_id % 5 == 0:
            logger.info(f"Processed {fold_id}/{len(folds)} folds")

    perf_long = pd.DataFrame(perf_rows)
    perf_long.to_csv(out_dir / "perf_lowdata_long.csv", index=False, encoding="utf-8-sig")

    perf_mean_sd = (perf_long
                    .groupby(["method", "k", "frac_real", "train_mode", "clf"])
                    .agg(
                        auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
                        bacc_mean=("bacc", "mean"), bacc_sd=("bacc", "std"),
                        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
                        logloss_mean=("logloss", "mean"), logloss_sd=("logloss", "std"),
                        n_train_real_mean=("n_train_real", "mean"),
                        n_train_total_mean=("n_train_total", "mean"),
                        n_val_mean=("n_val", "mean"),
                    )
                    .reset_index())
    perf_mean_sd.to_csv(out_dir / "perf_lowdata_mean_sd.csv", index=False, encoding="utf-8-sig")

    logger.info(f"DONE. Output folder: {out_dir}")


if __name__ == "__main__":
    main()
