#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_single_split_augmentation.py

Single-split augmentation workflow:
1) Load Saline and METH brain-region data from Excel.
2) Apply a fixed stratified split: Saline (8 train, 3 test) and METH (9 train, 4 test).
3) Train group-specific generators using the training subset only and the implementation in graph_mdwgan.py.
4) Evaluate 2x/4x/6x/8x/10x/12x augmentation factors using MMD, JS divergence, and SWD.
5) Select the most stable factor and export:
   - train_augmented_x{BEST}.csv (real training samples plus synthetic samples)
   - test_original.csv (held-out real samples)
   - factor_metrics.csv and splits_and_choice.json for reproducibility.

This script depends on scripts/graph_mdwgan.py and does not modify its model architecture or training hyperparameters.
"""

import os
import json
import argparse
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch

# ---- Import the reference Graph-MDWGAN implementation ----
from graph_mdwgan import (
    load_excel_data,                # Excel -> (Xs, Xm, roi_names)
    train_group,                    # Group-specific training with internal early stopping and evaluation
    postprocess_v10k,               # v10k post-processing
    pack_scores,                    # Return evaluation metrics including mmd_rbf and js_avg
    sliced_wasserstein_distance,    # Sliced Wasserstein distance
    EPS_POS
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DUP_DECIMALS = 6  # Decimal precision used for duplicate-sample matching


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============== Restore the generator from a checkpoint and sample ==============
def _load_generator_from_ckpt(ckpt_path: str, device: str = DEVICE):
    """
    Restore generator weights from a graph_mdwgan.py checkpoint. EMA weights are preferred when available.
    Returns: (G, Ahat_tensor, scaler_mean_tensor, scaler_scale_tensor, model_cfg).
    """
    from graph_mdwgan import Generator  # Use the reference implementation
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

    # Load trained generator weights
    G.load_state_dict(ck["G"])
    G.eval()

    # Replace parameters with EMA weights when available for stable inference
    if "ema" in ck and isinstance(ck["ema"], dict):
        with torch.no_grad():
            for n, p in G.named_parameters():
                if p.requires_grad and (n in ck["ema"]):
                    arr = ck["ema"][n]
                    tensor = torch.tensor(arr, device=device, dtype=p.dtype)
                    p.data.copy_(tensor)

    # Graph adjacency and scaling parameters
    Ahat = torch.tensor(ck["Ahat"], device=device, dtype=torch.float32)
    scaler_mean = torch.tensor(ck["scaler_mean"], device=device, dtype=torch.float32)
    scaler_scale = torch.tensor(ck["scaler_scale"], device=device, dtype=torch.float32)
    return G, Ahat, scaler_mean, scaler_scale, cfg


@torch.no_grad()
def _generate_synthetic(G, Ahat, scaler_mean, scaler_scale, n_samples: int, batch_size: int = 128, device: str = DEVICE):
    """
    Generate n_samples observations with the trained generator and return values in the pre-v10k domain.
    Workflow: standardized log domain -> inverse scaling -> expm1 to the original count domain.
    """
    # Infer the latent noise dimension
    if hasattr(G, "fc1") and hasattr(G.fc1, "in_features"):
        Z_DIM = G.fc1.in_features
    else:
        Z_DIM = 128  # Fallback for checkpoints without an explicit input dimension

    out_list = []
    remain = int(n_samples)
    while remain > 0:
        b = min(batch_size, remain)
        z = torch.randn(b, Z_DIM, device=device)
        fk_std = G(z, Ahat)                     # 标准化后的 log 域
        fk_log = fk_std * scaler_scale + scaler_mean
        fk_before = torch.expm1(fk_log).clamp_min(EPS_POS)
        out_list.append(fk_before.detach().cpu().numpy().astype(np.float32))
        remain -= b
    return np.vstack(out_list) if out_list else np.zeros((0, int(scaler_mean.shape[0])), dtype=np.float32)


def _metrics_mmd_js_swd(real_np: np.ndarray, synth_np: np.ndarray, decimals: int = DUP_DECIMALS):
    """
    与训练评估一致：
    - 先对合成样本做 v10k 后处理到目标分布；
    - 转 log 域后，计算 pack_scores（取 mmd_rbf, js_avg）；
    - SWD 用 sliced_wasserstein_distance 计算（n_projections=64）。
    新增：
    - dup_to_real：合成样本中与真实样本重复的比例（按 round(decimals) 判断）
    - dup_within：合成样本内部重复比例（按 round(decimals) 判断）
    """
    # v10k post-processing
    fk_after_np = postprocess_v10k(real_np, synth_np, final_export=False)

    # ===== 重复度（在 v10k 后处理之后计算，保证口径一致）=====
    if fk_after_np.shape[0] == 0:
        dup_to_real = float("nan")
        dup_within = float("nan")
    else:
        real_r = np.round(real_np.astype(np.float64), decimals=decimals)
        syn_r = np.round(fk_after_np.astype(np.float64), decimals=decimals)

        real_set = set(map(tuple, real_r))
        syn_tuples = list(map(tuple, syn_r))

        hit = sum(1 for row in syn_tuples if row in real_set)
        dup_to_real = float(hit / (len(syn_tuples) + 1e-12))

        uniq = len(set(syn_tuples))
        dup_within = float(1.0 - uniq / (len(syn_tuples) + 1e-12))

    # ===== MMD / JS（log 域）=====
    real_log = torch.tensor(np.log1p(np.clip(real_np, 0, None)).astype(np.float32))
    fk_after_log = torch.tensor(np.log1p(np.clip(fk_after_np, 0, None)).astype(np.float32))

    met = pack_scores(real_log, fk_after_log)
    mmd = float(met["mmd_rbf"])
    js = float(met["js_avg"])

    # ===== SWD：数量不一致时下采样到 min(n) =====
    n_real = real_log.shape[0]
    n_synth = fk_after_log.shape[0]

    if n_real != n_synth:
        min_samples = min(n_real, n_synth)
        real_indices = torch.randperm(n_real)[:min_samples]
        synth_indices = torch.randperm(n_synth)[:min_samples]
        real_log_sampled = real_log[real_indices]
        fk_after_log_sampled = fk_after_log[synth_indices]
    else:
        real_log_sampled = real_log
        fk_after_log_sampled = fk_after_log

    swd = float(sliced_wasserstein_distance(fk_after_log_sampled, real_log_sampled, n_projections=64).item())

    return {"mmd": mmd, "js": js, "swd": swd, "dup_to_real": dup_to_real, "dup_within": dup_within}


def _rank_select(best_table):
    """
    输入：{factor: {"mmd":v, "js":v, "swd":v}}
    策略：对每个指标做升序排名，求和取最小；若并列，按 mmd 较小者优先。
    返回：(best_factor, rank_sum_map)
    """
    factors = sorted(best_table.keys())

    def ranks_for(metric):
        vals = [(f, best_table[f][metric]) for f in factors]
        vals_sorted = sorted(vals, key=lambda x: x[1])  # 小优
        rank_map = {}
        i = 0
        while i < len(vals_sorted):
            j = i
            while j + 1 < len(vals_sorted) and np.isclose(vals_sorted[j+1][1], vals_sorted[i][1], atol=1e-12):
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j+1):
                rank_map[vals_sorted[k][0]] = avg_rank
            i = j + 1
        return rank_map

    r_m = ranks_for("mmd"); r_j = ranks_for("js"); r_s = ranks_for("swd")
    score = {f: (r_m[f] + r_j[f] + r_s[f]) for f in factors}
    best = sorted(factors, key=lambda f: (score[f], best_table[f]["mmd"]))[0]
    return best, score


# ============== 导出工具 ==============
def _make_rows(group_name: str, X: np.ndarray, roi_names, kind: str, start_idx: int = 0):
    rows: List[Dict[str, Union[str, float]]] = []
    for i, arr in enumerate(X):
        row: Dict[str, Union[str, float]] = {
            "Group": group_name,
            "Origin": kind,
            "SampleID": f"{group_name}_{kind}_{start_idx + i + 1:03d}"
        }
        for j, roi in enumerate(roi_names):
            row[str(roi)] = float(arr[j])
        rows.append(row)
    return rows


# ============== 主流程 ==============
def main():
    ap = argparse.ArgumentParser(description="Split data, train grouped GAN, test 2/4/6/8/10/12× augmentation, pick the most stable, and export two files.")
    ap.add_argument("--excel", type=str, default="data/brain_region_cfos.xlsx", help="脑区_填充后.xlsx 路径")
    ap.add_argument("--sheet", type=str, default=None, help="Excel 工作表名（可选）")
    ap.add_argument("--seed", type=int, default=20250829)

    # Fixed group-specific train/test counts
    ap.add_argument("--saline_train", type=int, default=8)
    ap.add_argument("--saline_test",  type=int, default=3)
    ap.add_argument("--meth_train",   type=int, default=9)
    ap.add_argument("--meth_test",    type=int, default=4)

    # 扩增倍数候选
    ap.add_argument("--factors", type=str, default="2,4,6,8,10,12", help="逗号分隔的扩增倍数（至少2），如 2,4,6,8,10,12")

    # 目录
    ap.add_argument("--out_dir", type=str, default="split_aug_outputs", help="输出目录（csv）")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints", help="训练权重目录（与主脚本保持一致）")

    # 可选导出为 Excel
    ap.add_argument("--export_excel", action="store_true", help="同时导出为 .xlsx 文件")

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 读数据
    Xs, Xm, roi_names = load_excel_data(args.excel, args.sheet)
    nS, nM = len(Xs), len(Xm)

    # 2) 检查并划分（分组独立，按种子可复现）
    assert args.saline_train + args.saline_test <= nS, f"Saline样本不足: {nS}"
    assert args.meth_train   + args.meth_test  <= nM, f"METH样本不足: {nM}"

    rngS = np.random.RandomState(args.seed).permutation(nS)
    rngM = np.random.RandomState(args.seed + 1).permutation(nM)

    S_train_idx = rngS[:args.saline_train]
    S_test_idx  = rngS[args.saline_train: args.saline_train + args.saline_test]
    M_train_idx = rngM[:args.meth_train]
    M_test_idx  = rngM[args.meth_train: args.meth_train + args.meth_test]

    Xs_train = Xs[S_train_idx]
    Xs_test  = Xs[S_test_idx]
    Xm_train = Xm[M_train_idx]
    Xm_test  = Xm[M_test_idx]

    # 准备 Ahat_np (通常使用完整数据集计算协方差矩阵)
    # 合并所有数据来计算全局协方差矩阵
    X_all = np.vstack([Xs, Xm])
    X_all_log = np.log1p(np.clip(X_all, 0, None))
    Ahat_np = np.corrcoef(X_all_log.T).astype(np.float32)  # 或使用 np.cov

    # 检查最终 ckpt 是否存在
    ckpt_S = os.path.join(args.ckpt_dir, "Saline", f"best_seed{args.seed}.pt")
    ckpt_M = os.path.join(args.ckpt_dir, "METH",   f"best_seed{args.seed + 1}.pt")
    
    # 3) 训练两个组（仅训练集；保留原超参与逻辑）
    if os.path.exists(ckpt_S):
        print(f"\n=== 发现已有 Saline 模型，跳过训练：{ckpt_S} ===")
    else:
        print("\n=== 训练 Saline 组生成器（仅训练集） ===")
        best_S, scaler_S = train_group("Saline", Xs_train, Ahat_np, out_dir=args.ckpt_dir, seed=args.seed)
    
    if os.path.exists(ckpt_M):
        print(f"\n=== 发现已有 METH 模型，跳过训练：{ckpt_M} ===")
    else:
        print("\n=== 训练 METH 组生成器（仅训练集） ===")
        best_M, scaler_M = train_group("METH", Xm_train, Ahat_np, out_dir=args.ckpt_dir, seed=args.seed + 1)

    if not os.path.exists(ckpt_S):
        raise FileNotFoundError(f"未找到 Saline 最终模型: {ckpt_S}")
    if not os.path.exists(ckpt_M):
        raise FileNotFoundError(f"未找到 METH 最终模型: {ckpt_M}")

    # 4) 评估 2×/4×/6×/8×/10×/12× 倍数并择优
    # 解析 factors
    factors = [int(x.strip()) for x in args.factors.split(",") if x.strip()]
    factors = sorted(list(set([f for f in factors if f >= 2])))
    if not factors:
        raise ValueError("有效的扩增倍数为空（至少包含 >=2 的整数）。")

    # 预加载生成器
    GS, AhatS, meanS, scaleS, cfgS = _load_generator_from_ckpt(ckpt_S, DEVICE)
    GM, AhatM, meanM, scaleM, cfgM = _load_generator_from_ckpt(ckpt_M, DEVICE)

    results = {}       # {factor: {"mmd":..., "js":..., "swd":...}}
    synth_cache = {}   # {factor: {"Saline": np.ndarray, "METH": np.ndarray}}

    nS_tr, nM_tr = len(Xs_train), len(Xm_train)

    for f in factors:
        print(f"\\n>>> 评估扩增倍数：×{f}")
        # 每组合成样本数量（最终训练集规模 = 原始训练 + 合成）
        gen_S = nS_tr * (f - 1)
        gen_M = nM_tr * (f - 1)

        # 生成（BEFORE 域；评估内部做 v10k）
        synth_S = _generate_synthetic(GS, AhatS, meanS, scaleS, gen_S, device=DEVICE)
        synth_M = _generate_synthetic(GM, AhatM, meanM, scaleM, gen_M, device=DEVICE)

        # 分别对比各自组的训练实测，计算指标
        met_S = _metrics_mmd_js_swd(Xs_train, synth_S)
        met_M = _metrics_mmd_js_swd(Xm_train, synth_M)

        # 按训练样本数加权平均指标（新增 dup_to_real / dup_within）
        agg = {}
        for k in ("mmd", "js", "swd", "dup_to_real", "dup_within"):
            agg[k] = (met_S[k] * nS_tr + met_M[k] * nM_tr) / float(nS_tr + nM_tr)
        results[f] = agg
        synth_cache[f] = {"Saline": synth_S, "METH": synth_M}

        print(f"Saline: MMD={met_S['mmd']:.6f}, JS={met_S['js']:.6f}, SWD={met_S['swd']:.6f}, "
              f"dupR={met_S['dup_to_real']:.4f}, dupA={met_S['dup_within']:.4f}")
        print(f"METH  : MMD={met_M['mmd']:.6f}, JS={met_M['js']:.6f}, SWD={met_M['swd']:.6f}, "
              f"dupR={met_M['dup_to_real']:.4f}, dupA={met_M['dup_within']:.4f}")
        print(f"[Weighted summary] MMD={agg['mmd']:.6f}, JS={agg['js']:.6f}, SWD={agg['swd']:.6f}, "
              f"dupR={agg['dup_to_real']:.4f}, dupA={agg['dup_within']:.4f}")

    best_factor, rank_scores = _rank_select(results)
    print("\\n=== Augmentation-factor selection ===")
    for f in factors:
        print(f"x{f}: rank_sum={rank_scores[f]:.2f}  (MMD={results[f]['mmd']:.6f}, JS={results[f]['js']:.6f}, SWD={results[f]['swd']:.6f})")
    print(f"\\nSelected augmentation factor: x{best_factor}")

    # Save factor-comparison metrics
    df_metrics = pd.DataFrame(
        [{"factor": f, **results[f], "rank_sum": rank_scores[f]} for f in factors]
    ).sort_values("factor")
    df_metrics.to_csv(os.path.join(args.out_dir, "factor_metrics.csv"), index=False)

    # 5) Export augmented training data and held-out test data
    synth_best_S = synth_cache[best_factor]["Saline"]
    synth_best_M = synth_cache[best_factor]["METH"]

    # Training set: real training samples plus group-specific synthetic samples
    train_rows = []
    train_rows += _make_rows("Saline", Xs_train, roi_names, "RealTrain")
    train_rows += _make_rows("METH",   Xm_train, roi_names, "RealTrain")
    train_rows += _make_rows("Saline", synth_best_S, roi_names, "Synthetic")
    train_rows += _make_rows("METH",   synth_best_M, roi_names, "Synthetic")

    # Test set: held-out real samples only
    test_rows = []
    test_rows += _make_rows("Saline", Xs_test, roi_names, "RealTest")
    test_rows += _make_rows("METH",   Xm_test, roi_names, "RealTest")

    df_train = pd.DataFrame(train_rows)
    df_test  = pd.DataFrame(test_rows)

    out_train_csv = os.path.join(args.out_dir, f"train_augmented_x{best_factor}.csv")
    out_test_csv  = os.path.join(args.out_dir, "test_original.csv")
    df_train.to_csv(out_train_csv, index=False)
    df_test.to_csv(out_test_csv, index=False)

    # Optional Excel export
    if args.export_excel:
        out_train_xlsx = os.path.join(args.out_dir, f"train_augmented_x{best_factor}.xlsx")
        out_test_xlsx  = os.path.join(args.out_dir, "test_original.xlsx")
        with pd.ExcelWriter(out_train_xlsx, engine="xlsxwriter") as writer:
            df_train.to_excel(writer, sheet_name="train", index=False)
        with pd.ExcelWriter(out_test_xlsx, engine="xlsxwriter") as writer:
            df_test.to_excel(writer, sheet_name="test", index=False)

    # Save split indices and factor-selection metadata
    split_rec = {
        "seed": args.seed,
        "roi_names": list(map(str, roi_names)),
        "Saline": {"train_index": S_train_idx.tolist(), "test_index": S_test_idx.tolist()},
        "METH":   {"train_index": M_train_idx.tolist(), "test_index": M_test_idx.tolist()},
        "chosen_factor": int(best_factor),
        "metrics": results
    }
    with open(os.path.join(args.out_dir, "splits_and_choice.json"), "w", encoding="utf-8") as f:
        json.dump(split_rec, f, ensure_ascii=False, indent=2)

    print("\\nGenerated outputs:")
    print(f"  - {out_train_csv}")
    print(f"  - {out_test_csv}")
    print(f"  - {os.path.join(args.out_dir, 'factor_metrics.csv')}")
    print(f"  - {os.path.join(args.out_dir, 'splits_and_choice.json')}")
    if args.export_excel:
        print("  - Corresponding .xlsx files")
    print("Note: augmentation is restricted to the training set and performed separately for Saline and METH using the reference training configuration.")


if __name__ == "__main__":
    main()
