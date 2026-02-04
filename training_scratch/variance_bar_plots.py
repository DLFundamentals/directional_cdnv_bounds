#!/usr/bin/env python3
import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_scratch.data.mini_imagenet_datamodule import MiniImageNetCfg, MiniImageNetDataModule
from training_scratch.utils.eval_utils import (find_checkpoint_files, extract_features,
                                              extract_backbone_features, load_model_from_checkpoint)

def compute_class_cache(X, y, max_per_class=None, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    Xc = {}
    mu = {}
    vtot = {}  # total within-class variance (mean squared norm of residual)
    for c in classes:
        Z = X[y == c]
        if max_per_class is not None and Z.shape[0] > max_per_class:
            idx = rng.choice(Z.shape[0], size=max_per_class, replace=False)
            Z = Z[idx]
        m = Z.mean(axis=0)
        R = Z - m
        Xc[int(c)] = Z
        mu[int(c)] = m
        vtot[int(c)] = float(np.mean(np.sum(R * R, axis=1)))
    return classes.astype(int), Xc, mu, vtot

def rank_pairs_by_dircdnv(Xc, mu, vtot, classes, top_k=20):
    """
    Returns list of dicts with metrics for each pair.
    Complexity ~ O(#pairs * (ni+nj)*d) for projection vars; feasible for C~100 with subsampling.
    """
    results = []
    C = len(classes)
    for a in range(C):
        i = int(classes[a])
        Zi = Xc[i]; mui = mu[i]; Ri = Zi - mui
        for b in range(a+1, C):
            j = int(classes[b])
            Zj = Xc[j]; muj = mu[j]; Rj = Zj - muj

            dvec = muj - mui
            d = np.linalg.norm(dvec)
            if d < 1e-12:
                continue
            u = dvec / d
            d2 = d * d
            # directional variances along u
            pi = Ri @ u
            pj = Rj @ u
            var_i = float(np.var(pi, ddof=0))
            var_j = float(np.var(pj, ddof=0))

            dir_cdnv_sym = 0.5 * (var_i + var_j) / (d2 + 1e-12)
            cdnv = (vtot[i] + vtot[j]) / (d2 + 1e-12)

            # "drama" = how much more total variance than decision-axis variance
            # equivalent to CDNV / dirCDNV(sym) up to constants
            drama = cdnv / (dir_cdnv_sym + 1e-12)

            results.append({
                "i": i, "j": j,
                "d": float(d),
                "cdnv": float(cdnv),
                "dir_cdnv_sym": float(dir_cdnv_sym),
                "drama": float(drama),
            })

    # First, get lowest dir-CDNV
    results.sort(key=lambda r: r["dir_cdnv_sym"])
    low_dir = results[:max(top_k, 50)]  # take a pool
    # Then pick the most dramatic within that pool
    low_dir_sorted_by_drama = sorted(low_dir, key=lambda r: (-r["drama"], r["dir_cdnv_sym"]))
    return results, low_dir_sorted_by_drama

def pair_variance_decomposition(X, Y, i, j, k_list=(5, 10, 100)):
    Xi = X[Y == i]
    Xj = X[Y == j]

    mu_i = Xi.mean(axis=0)
    mu_j = Xj.mean(axis=0)

    u = mu_j - mu_i
    u = u / (np.linalg.norm(u) + 1e-12)

    Ri = Xi - mu_i
    Rj = Xj - mu_j
    R = np.concatenate([Ri, Rj], axis=0)
    N = R.shape[0]

    ru = R @ u
    var_u = np.mean(ru ** 2)

    R_perp = R - ru[:, None] * u[None, :]
    _, S, _ = np.linalg.svd(R_perp, full_matrices=False)
    lam = (S ** 2) / N

    out = {"var_u": var_u, "var_orth_total": lam.sum()}
    for k in k_list:
        out[f"var_orth_top{k}"] = lam[:k].sum()
    return out

def collect_pair_over_epochs(var_decomp_per_epoch, pairs, epochs, k_list=(5, 20, 100)):
    series = []  
    for ep in epochs:
        for r in pairs[:10]:
            i, j = r['i'], r['j']
            d = var_decomp_per_epoch[ep][(i, j)]
            row = {
                "epoch": ep,
                "pair": (i, j),
                "var_u": d["var_u"],
                "var_orth_total": d["var_orth_total"],
            }
            for k in k_list:
                row[f"var_orth_top{k}"] = d[f"var_orth_top{k}"]
            series.append(row)
    return series

def summarize_rows(rows, k_list=(5, 20, 100)):
    """
    Produces:
      summary[epoch][metric] = (mean, std)
    """
    metrics = ["var_u"] + [f"var_orth_top{k}" for k in k_list] + ["var_orth_total"]

    bucket = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ep = r["epoch"]
        for m in metrics:
            bucket[ep][m].append(r[m])

    summary = {}
    for ep in sorted(bucket.keys()):
        summary[ep] = {}
        for m in metrics:
            arr = np.asarray(bucket[ep][m], dtype=float)
            summary[ep][m] = (float(arr.mean()), float(arr.std(ddof=0)))
    return summary

def main():
    parser = argparse.ArgumentParser(description='Batch CDNV evaluation across checkpoints')
    parser.add_argument('--ckpt_dir', required=True, help='Directory containing checkpoint files')
    parser.add_argument('--out_csv', default='cdnv_by_epoch.csv', help='Output CSV file path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--start', type=int, default=0, help='Start epoch')
    parser.add_argument('--end', type=int, default=1000, help='End epoch')
    parser.add_argument('--max_train_batches', type=int, default=200, help='Max batches for train features')
    parser.add_argument('--max_val_batches', type=int, default=50, help='Max batches for val features')
    args = parser.parse_args()

    device = torch.device(args.device)
    # Find checkpoint files
    files = find_checkpoint_files(args.ckpt_dir, start=args.start, end=args.end)
    if len(files) == 0:
        raise FileNotFoundError(f'No checkpoint files found in {args.ckpt_dir} for range [{args.start}, {args.end}]')
    
    print(f"Found {len(files)} checkpoint files")
    
    # Load first checkpoint to get config for datamodule
    first_ckpt_path = files[0][1]
    first_ckpt = torch.load(first_ckpt_path, map_location='cpu')
    hparams = first_ckpt['hyper_parameters']
    cfg = OmegaConf.create(hparams)
    
    # Build datamodule from checkpoint config
    data_cfg = MiniImageNetCfg(**cfg.data)
    dm = MiniImageNetDataModule(data_cfg)
    dm.setup()
    
    rows = []
    epochs = [0, 10, 100, 1000]
    var_decomp_per_epoch = defaultdict(dict)
    for epoch, ckpt_path in files[::-1]:  # process in reverse order (latest first)
        if epoch not in epochs:
            continue
        print(f'Processing checkpoint: {ckpt_path} (epoch={epoch})')
        try:
            # Load model from checkpoint
            model, model_cfg = load_model_from_checkpoint(ckpt_path, device)
            model.eval()
            model.to(device)
            backbone = model.backbone
            
            # Extract features
            train_loader = dm.probe_train_dataloader() if hasattr(dm, "probe_train_dataloader") else dm.train_dataloader()
            val_loader = dm.probe_test_dataloader() if hasattr(dm, "probe_test_dataloader") else dm.val_dataloader()
            
            Xtr, Ytr = extract_features(train_loader, backbone, device, max_batches=args.max_train_batches)
            Xva, Yva = extract_features(val_loader, backbone, device, max_batches=args.max_val_batches)
            
            print(f"  Train features: {Xtr.shape}, Val features: {Xva.shape}")
            
            # Ensure numpy arrays
            Xtr_np = Xtr.detach().cpu().numpy() if hasattr(Xtr, "detach") else np.asarray(Xtr)
            Ytr_np = Ytr.detach().cpu().numpy() if hasattr(Ytr, "detach") else np.asarray(Ytr)
            Xva_np = Xva.detach().cpu().numpy() if hasattr(Xva, "detach") else np.asarray(Xva)
            Yva_np = Yva.detach().cpu().numpy() if hasattr(Yva, "detach") else np.asarray(Yva)
        
            # get best pairs from epoch 0 and keep them consistent across epochs
            if epoch == 1000:
                # ---- run ranking ----
                classes, Xc, mu, vtot = compute_class_cache(
                    Xtr_np, Ytr_np,
                    max_per_class=500,     # bump to 1000 if you can, 500 is usually enough
                    seed=getattr(args, "seed", 0)
                )

                all_pairs, best_pairs = rank_pairs_by_dircdnv(Xc, mu, vtot, classes, top_k=50)

            print("Top candidates (low dir-CDNV, high drama):")

            for n, r in enumerate(best_pairs[:10]):
                print(f"({r['i']},{r['j']})  d={r['d']:.4f}  dir={r['dir_cdnv_sym']:.6f}  CDNV={r['cdnv']:.3f}  drama={r['drama']:.1f}")

                i, j = r['i'], r['j']
                # Compute axes from TRAIN (recommended), then visualize both train & val in same basis
                var_decomp_per_epoch[epoch][(i, j)] = pair_variance_decomposition(Xtr_np, Ytr_np, i, j,
                                                                                   k_list=(1, 10, 100))

            # Clean up to save memory
            del model, backbone, Xtr, Ytr, Xva, Yva
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            import traceback
            print(f'ERROR processing {ckpt_path}: {e}')
            traceback.print_exc()
            rows.append({'epoch': epoch, 'error': str(e)})
    

    series = collect_pair_over_epochs(var_decomp_per_epoch, best_pairs, epochs, k_list=(1, 10, 100))
    summary = summarize_rows(series, k_list=(1, 10, 100))

    # save summary
    cols = ['epoch', 'var_u_mean', 'var_u_std'] + \
           [f'var_orth_top{k}_mean' for k in (1, 10, 100)] + \
           [f'var_orth_top{k}_std' for k in (1, 10, 100)] + \
           ['var_orth_total_mean', 'var_orth_total_std']
    summary_rows = []
    for ep in epochs:
        row = {'epoch': ep}
        for metric in ['var_u'] + [f'var_orth_top{k}' for k in (1, 10, 100)] + ['var_orth_total']:
            mean, std = summary[ep][metric]
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f"./variance_viz/summary_pair_top10.csv", index=False)

if __name__ == '__main__':
    main()