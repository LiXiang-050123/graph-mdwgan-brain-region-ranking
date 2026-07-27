# run_5seeds_full_pipeline_x8.py
# 一键跑 5 个 seed：Ours(重新划分+扩增x8) + ROSx8 + SMOTEx8，并汇总 mean±sd
# 依赖：pandas numpy torch scikit-learn
# 要求：repository scripts directory contains run_single_split_augmentation.py and graph_mdwgan.py

import os
import sys
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from graph_mdwgan import pack_scores, sliced_wasserstein_distance

# ====== 你的原始数据（已写死）======
FULL_EXCEL = "data/brain_region_cfos.xlsx"
# ====== 配置 ======
SEEDS = [20250829, 20250830, 20250831, 20250832, 20250833]
FACTOR = 8
ROOT_DIR = Path("outputs/quality_5seeds")
DUP_DECIMALS = 6
K_NEIGHBORS = 5
N_PROJ = 64

META_COLS = {"Group", "Origin", "SampleID"}


def run_cmd(cmd_list):
    print("\n>>", " ".join(str(x) for x in cmd_list))
    subprocess.check_call(cmd_list)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def feat_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in META_COLS]


def metrics_mmd_js_swd_no_v10k(real_np: np.ndarray, synth_np: np.ndarray, seed: int):
    """
    ROS/SMOTE：跳过 v10k，直接计算 log1p + pack_scores(mmd_rbf/js_avg) + SWD
    """
    real_log = torch.tensor(np.log1p(np.clip(real_np, 0, None)).astype(np.float32))
    syn_log  = torch.tensor(np.log1p(np.clip(synth_np, 0, None)).astype(np.float32))

    met = pack_scores(real_log, syn_log)
    mmd = float(met["mmd_rbf"])
    js  = float(met["js_avg"])

    n = min(real_log.shape[0], syn_log.shape[0])
    if real_log.shape[0] != syn_log.shape[0]:
        torch.manual_seed(seed)
        real_log = real_log[torch.randperm(real_log.shape[0])[:n]]
        syn_log  = syn_log[torch.randperm(syn_log.shape[0])[:n]]

    swd = float(sliced_wasserstein_distance(syn_log, real_log, n_projections=N_PROJ).item())
    return mmd, js, swd


def dup_rates(real_np: np.ndarray, added_np: np.ndarray, decimals: int = DUP_DECIMALS):
    """
    dup_to_real：新增样本中，与真实样本重复的比例
    dup_within ：新增样本内部重复比例
    浮点用 round(decimals) 作为容差
    """
    if added_np.shape[0] == 0:
        return float("nan"), float("nan")

    real_r = np.round(real_np.astype(np.float64), decimals)
    add_r  = np.round(added_np.astype(np.float64), decimals)

    real_set = set(map(tuple, real_r))
    add_tuples = list(map(tuple, add_r))

    dup_to_real = sum(1 for r in add_tuples if r in real_set) / (len(add_tuples) + 1e-12)
    dup_within  = 1.0 - (len(set(add_tuples)) / (len(add_tuples) + 1e-12))
    return float(dup_to_real), float(dup_within)


def make_ros(df_real: pd.DataFrame, factor: int, seed: int):
    rng = np.random.RandomState(seed)
    out = [df_real.copy()]
    for g, sub in df_real.groupby("Group"):
        n_add = len(sub) * (factor - 1)
        add = sub.sample(n=n_add, replace=True, random_state=rng.randint(0, 10**9)).copy()
        add["Origin"] = "ROS"
        add["SampleID"] = [f"{g}_ROS_{i+1:05d}" for i in range(len(add))]
        out.append(add)
    df_all = pd.concat(out, ignore_index=True)
    df_added = df_all[df_all["Origin"].astype(str) == "ROS"].copy()
    return df_all, df_added


def smote_generate_group(X: np.ndarray, n_add: int, seed: int, k: int = K_NEIGHBORS):
    if n_add <= 0:
        return np.zeros((0, X.shape[1]), float)

    n = X.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        raise ValueError(f"SMOTE 需要每组至少 2 个样本，但该组样本数={n}。")

    rng = np.random.RandomState(seed)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xs)
    neigh = np.asarray(nn.kneighbors(Xs, return_distance=False))[:, 1:]  # 去掉自身

    syn = np.empty((n_add, X.shape[1]), float)
    for t in range(n_add):
        i = rng.randint(0, n)
        j = rng.choice(neigh[i])
        lam = rng.rand()
        syn[t] = Xs[i] + lam * (Xs[j] - Xs[i])

    syn = scaler.inverse_transform(syn)
    syn = np.clip(syn, 0, None)
    return syn


def make_smote(df_real: pd.DataFrame, factor: int, seed: int):
    feats = feat_cols(df_real)
    adds = []
    for gi, (g, sub) in enumerate(df_real.groupby("Group")):
        X = sub[feats].to_numpy(float)
        n_add = len(sub) * (factor - 1)
        Xsyn = smote_generate_group(X, n_add=n_add, seed=seed + 100 + gi, k=K_NEIGHBORS)

        add = pd.DataFrame(Xsyn, columns=feats)
        add["Group"] = g
        add["Origin"] = "SMOTE"
        add["SampleID"] = [f"{g}_SMOTE_{i+1:05d}" for i in range(len(add))]
        adds.append(add)

    df_added = pd.concat(adds, ignore_index=True) if adds else pd.DataFrame(columns=df_real.columns)
    df_all = pd.concat([df_real, df_added[df_real.columns]], ignore_index=True)
    return df_all, df_added


def weighted_metrics(df_real: pd.DataFrame, df_added: pd.DataFrame, seed: int):
    """
    按组计算，再按该组真实样本数加权平均
    返回：mmd, js, swd, dup_to_real, dup_within
    """
    feats = feat_cols(df_real)
    groups = sorted(df_real["Group"].unique())
    weights = []
    arr = []

    for gi, g in enumerate(groups):
        R = df_real[df_real["Group"] == g][feats].to_numpy(float)
        A = df_added[df_added["Group"] == g][feats].to_numpy(float)

        mmd, js, swd = metrics_mmd_js_swd_no_v10k(R, A, seed=seed + 10 + gi)
        dtr, dwi = dup_rates(R, A, decimals=DUP_DECIMALS)

        weights.append(len(R))
        arr.append((mmd, js, swd, dtr, dwi))

    W = np.array(weights, float)
    W = W / (W.sum() + 1e-12)
    arr = np.array(arr, float)
    out = (arr * W[:, None]).sum(axis=0)
    return tuple(float(x) for x in out)


def weighted_dup_for_ours(df_real: pd.DataFrame, df_syn: pd.DataFrame):
    """
    Ours 的重复度也按组加权（Synthetic vs RealTrain）
    """
    feats = feat_cols(df_real)
    groups = sorted(df_real["Group"].unique())
    weights = []
    dtrs = []
    dwis = []

    for g in groups:
        R = df_real[df_real["Group"] == g][feats].to_numpy(float)
        S = df_syn[df_syn["Group"] == g][feats].to_numpy(float)
        dtr, dwi = dup_rates(R, S, decimals=DUP_DECIMALS)
        weights.append(len(R))
        dtrs.append(dtr)
        dwis.append(dwi)

    W = np.array(weights, float)
    W = W / (W.sum() + 1e-12)
    return float((np.array(dtrs) * W).sum()), float((np.array(dwis) * W).sum())


def aggregate_all(root_dir: Path, out_name: str = "summary_mean_sd_x8.csv"):
    files = glob.glob(str(root_dir / "seed_*" / "compare" / "compare_x8_seed*.csv"))
    if not files:
        raise FileNotFoundError("未找到 compare_x8_seed*.csv，请确认单 seed 跑通后再汇总。")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    metric_cols = ["mmd", "js", "swd", "dup_to_real", "dup_within"]
    g = df.groupby("method")[metric_cols]
    mean = g.mean().add_suffix("_mean")
    sd = g.std(ddof=1).add_suffix("_sd")

    out = pd.concat([mean, sd], axis=1).reset_index()
    out.to_csv(root_dir / out_name, index=False, encoding="utf-8-sig")
    df.to_csv(root_dir / "all_seeds_long.csv", index=False, encoding="utf-8-sig")

    print("\nSaved summary:")
    print("  ", root_dir / out_name)
    print("Saved long table:")
    print("  ", root_dir / "all_seeds_long.csv")


def main():
    ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for s in SEEDS:
        set_seed(s)

        seed_dir = ROOT_DIR / f"seed_{s}"
        ours_dir = seed_dir / "ours"
        ckpt_dir = seed_dir / "ckpt"
        cmp_dir  = seed_dir / "compare"
        ros_dir  = seed_dir / "ros"
        sm_dir   = seed_dir / "smote"

        for d in [ours_dir, ckpt_dir, cmp_dir, ros_dir, sm_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # ===== 1) 跑 Ours：每个 seed 重新划分（固定 factor=8，保证可比）=====
        run_cmd([
            sys.executable, "scripts/run_single_split_augmentation.py",
            "--excel", FULL_EXCEL,
            "--seed", str(s),
            "--factors", str(FACTOR),
            "--out_dir", str(ours_dir),
            "--ckpt_dir", str(ckpt_dir)
        ])

        train_aug_path = ours_dir / f"train_augmented_x{FACTOR}.csv"
        factor_met_path = ours_dir / "factor_metrics.csv"

        if not train_aug_path.exists():
            raise FileNotFoundError(f"未找到 {train_aug_path}，请确认 integrated_split_augment.py 的输出文件名。")
        if not factor_met_path.exists():
            raise FileNotFoundError(f"未找到 {factor_met_path}。")

        df_aug = pd.read_csv(train_aug_path)
        if "Origin" not in df_aug.columns:
            raise ValueError("train_augmented_x*.csv 缺少 Origin 列，无法区分 RealTrain/Synthetic。")

        df_real = df_aug[df_aug["Origin"].astype(str) == "RealTrain"].copy()
        df_syn  = df_aug[df_aug["Origin"].astype(str) == "Synthetic"].copy()

        if len(df_real) == 0:
            raise ValueError("未找到 Origin=RealTrain 的样本。")
        if len(df_syn) == 0:
            raise ValueError("未找到 Origin=Synthetic 的样本。")

        # Ours：mmd/js/swd 从 factor_metrics.csv 读取；dup 从 Synthetic vs RealTrain 计算
        fm = pd.read_csv(factor_met_path)
        row = fm[fm["factor"] == FACTOR]
        if len(row) == 0:
            raise ValueError(f"factor_metrics.csv 中未找到 factor={FACTOR}。")
        row = row.iloc[0]
        ours_mmd = float(row["mmd"])
        ours_js  = float(row["js"])
        ours_swd = float(row["swd"])
        ours_dtr, ours_dwi = weighted_dup_for_ours(df_real, df_syn)

        # ===== 2) ROS x8 =====
        df_ros_all, df_ros_added = make_ros(df_real, FACTOR, seed=s + 1)
        df_ros_all.to_csv(ros_dir / f"train_ros_x{FACTOR}.csv", index=False, encoding="utf-8-sig")
        ros_mmd, ros_js, ros_swd, ros_dtr, ros_dwi = weighted_metrics(df_real, df_ros_added, seed=s + 1000)

        pd.DataFrame([{
            "factor": FACTOR, "mmd": ros_mmd, "js": ros_js, "swd": ros_swd,
            "dup_to_real": ros_dtr, "dup_within": ros_dwi, "rank_sum": np.nan
        }]).to_csv(ros_dir / f"ros_factor_metrics_x{FACTOR}.csv", index=False, encoding="utf-8-sig")

        # ===== 3) SMOTE x8 =====
        df_sm_all, df_sm_added = make_smote(df_real, FACTOR, seed=s + 2)
        df_sm_all.to_csv(sm_dir / f"train_smote_x{FACTOR}.csv", index=False, encoding="utf-8-sig")
        sm_mmd, sm_js, sm_swd, sm_dtr, sm_dwi = weighted_metrics(df_real, df_sm_added, seed=s + 2000)

        pd.DataFrame([{
            "factor": FACTOR, "mmd": sm_mmd, "js": sm_js, "swd": sm_swd,
            "dup_to_real": sm_dtr, "dup_within": sm_dwi, "rank_sum": np.nan
        }]).to_csv(sm_dir / f"smote_factor_metrics_x{FACTOR}.csv", index=False, encoding="utf-8-sig")

        # ===== 4) 三方法对比表（该 seed）=====
        compare = pd.DataFrame([
            {"method": "Ours", "seed": s, "factor": FACTOR, "mmd": ours_mmd, "js": ours_js, "swd": ours_swd,
             "dup_to_real": ours_dtr, "dup_within": ours_dwi},
            {"method": "ROS(no v10k)", "seed": s, "factor": FACTOR, "mmd": ros_mmd, "js": ros_js, "swd": ros_swd,
             "dup_to_real": ros_dtr, "dup_within": ros_dwi},
            {"method": "SMOTE(no v10k)", "seed": s, "factor": FACTOR, "mmd": sm_mmd, "js": sm_js, "swd": sm_swd,
             "dup_to_real": sm_dtr, "dup_within": sm_dwi},
        ])

        out_cmp = cmp_dir / f"compare_x{FACTOR}_seed{s}.csv"
        compare.to_csv(out_cmp, index=False, encoding="utf-8-sig")
        print("Saved:", out_cmp)

    # ===== 5) 汇总 5 seed：mean±sd =====
    aggregate_all(ROOT_DIR, out_name=f"summary_mean_sd_x{FACTOR}.csv")


if __name__ == "__main__":
    main()
