#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Any
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

from eval_utils.geometry import GeometricEvaluator
from training_scratch.data.mini_imagenet_datamodule import MiniImageNetCfg, MiniImageNetDataModule


def find_checkpoint_files(ckpt_dir, start=0, end=1000):
    """Find all checkpoint files in the given directory within the epoch range."""
    files = []
    for epoch in range(start, end + 1):
        fname = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.ckpt")
        if os.path.exists(fname):
            files.append((epoch, fname))
    # Also include last.ckpt if present
    lastp = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(lastp):
        files.append(('last', lastp))
    # Sort: numeric epochs first, then 'last'
    files.sort(key=lambda x: (x[0] == 'last', x[0] if isinstance(x[0], int) else 9999))
    return files


def extract_backbone_features(backbone, images):
    """
    Extract features from backbone, handling both ViT and ResNet architectures.
    Returns features [B, D].
    """
    out = None
    
    # Handle models with nested `vit` attribute (some wrappers)
    if hasattr(backbone, 'vit'):
        vit = backbone.vit
        out = vit.forward_features(images)
    # Try common hooks used by various encoders
    elif hasattr(backbone, 'forward_features'):
        out = backbone.forward_features(images)
    else:
        # Direct call
        out = backbone(images)

    # Unwrap outputs
    feats = None
    if hasattr(out, 'last_hidden_state') or hasattr(out, 'pooler_output'):
        hidden = getattr(out, 'last_hidden_state', None)
        if hidden is not None:
            feats = hidden[:, 0] if hidden.dim() == 3 else hidden
        else:
            pooled = getattr(out, 'pooler_output', None)
            if pooled is not None:
                feats = pooled
    elif isinstance(out, dict):
        if 'last_hidden_state' in out:
            hidden = out['last_hidden_state']
            feats = hidden[:, 0] if hidden.dim() == 3 else hidden
        elif 'pooler_output' in out:
            feats = out['pooler_output']
    elif isinstance(out, torch.Tensor):
        if out.dim() == 3:
            feats = out[:, 0]
        elif out.dim() > 2:
            feats = torch.flatten(out, 1)
        else:
            feats = out

    if feats is None:
        raise RuntimeError("Could not extract features from backbone output")

    return feats


def extract_features(loader, backbone, device, max_batches=999999):
    """Extract features and labels from a dataloader."""
    feats_list, y_list = [], []

    for batch_idx, (views, y) in enumerate(tqdm(loader)):
        if batch_idx >= max_batches:
            break
        images = (
            views[0].to(device, non_blocking=True)
            if isinstance(views, (list, tuple))
            else views.to(device)
        )

        with torch.no_grad():
            feats = extract_backbone_features(backbone, images)
            feats = F.normalize(feats, dim=1)

        feats_list.append(feats.cpu())
        y_list.append(y.cpu())

    return torch.cat(feats_list, dim=0), torch.cat(y_list, dim=0)


def load_model_from_checkpoint(ckpt_path, device='cpu'):
    """
    Load a model from a PyTorch Lightning checkpoint.
    Uses the hyperparameters stored in the checkpoint itself.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    if 'hyper_parameters' not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} does not contain hyper_parameters")
    
    hparams = ckpt['hyper_parameters']
    cfg = OmegaConf.create(hparams)
    
    method_name = cfg.get('method', {}).get('name', '').lower()
    
    if method_name == 'vicreg':
        from training_scratch.models.vicreg import LightlyVICReg
        model = LightlyVICReg(cfg)
    elif method_name == 'mae':
        from training_scratch.models.mae import LightlyMAE
        model = LightlyMAE(cfg)
    elif method_name == 'dino':
        from training_scratch.models.dino import LightlyDINO
        model = LightlyDINO(cfg)
    elif method_name == 'ijepa':
        from training_scratch.models.ijepa import LightlyIJepa
        model = LightlyIJepa(cfg)
    else:
        state_dict = ckpt.get('state_dict', {})
        keys = list(state_dict.keys())
        is_vit = any('vit' in k or 'patch_embed' in k or 'mask_token' in k for k in keys)
        
        if is_vit:
            from training_scratch.models.mae import LightlyMAE
            model = LightlyMAE(cfg)
        else:
            from training_scratch.models.vicreg import LightlyVICReg
            model = LightlyVICReg(cfg)
    

    epoch = ckpt.get("epoch", None)
    if epoch != 0:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        print("⚠️ Skipping weight load for epoch 0 checkpoint")
    
    return model, cfg

def reinitialize_model(model):
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def pick_random_pair(labels: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes, got {len(classes)}.")
    i, j = rng.choice(classes, size=2, replace=False)
    return int(i), int(j)

def compute_pair_axes(
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    j: int,
    use_pair_only_for_within: bool = True,
):
    """
    Returns:
      u: decision axis (d,)
      v: top within-class variance axis orthogonal to u (d,)
      mu_i, mu_j: class means (d,)
      mid: midpoint (d,)
      mask_ij: boolean mask selecting only classes i or j
    """
    mask_ij = (y == i) | (y == j)
    if mask_ij.sum() < 4:
        raise ValueError(f"Not enough samples for pair ({i},{j}) to compute axes.")

    Xi = X[y == i]
    Xj = X[y == j]
    mu_i = Xi.mean(axis=0)
    mu_j = Xj.mean(axis=0)

    dvec = (mu_j - mu_i)
    dnorm = np.linalg.norm(dvec)
    if dnorm < 1e-12:
        raise ValueError(f"Means for classes ({i},{j}) are (nearly) identical; can't define u_ij.")
    u = dvec / dnorm

    # Midpoint centering for plotting coordinates
    mid = 0.5 * (mu_i + mu_j)

    # Build residual matrix R (within-class residuals)
    # For pairwise story: use only i and j residuals (recommended)
    if use_pair_only_for_within:
        Ri = Xi - mu_i
        Rj = Xj - mu_j
        R = np.concatenate([Ri, Rj], axis=0)  # [Nij, d]
    else:
        # pooled across all classes (sometimes smoother, but less "pair-pure")
        R = X - np.stack([ (X[y==c].mean(axis=0) if np.any(y==c) else 0) for c in np.unique(y) ], axis=0)[0]  # not used typically

    # Project residuals onto subspace orthogonal to u: P = I - uu^T
    # Implement projection as R_perp = R - (R u) u^T
    Ru = (R @ u)[:, None]          # [Nij, 1]
    R_perp = R - Ru * u[None, :]   # [Nij, d]

    # Top PC of R_perp gives v (direction of maximal variance orthogonal to u)
    # Use SVD for stability: R_perp = U S Vt, top right-singular vector = Vt[0]
    # Note: if R_perp is all ~0, then variance orthogonal to u is tiny; pick any orthonormal v.
    norm_Rp = np.linalg.norm(R_perp)
    if norm_Rp < 1e-12:
        # choose arbitrary orthonormal v
        # pick a basis vector not too aligned with u
        e = np.zeros_like(u)
        e[np.argmin(np.abs(u))] = 1.0
        v = e - (e @ u) * u
        v = v / (np.linalg.norm(v) + 1e-12)
    else:
        # SVD on centered residuals (already residual-centered)
        _, _, Vt = np.linalg.svd(R_perp, full_matrices=False)
        v = Vt[0]
        v = v / (np.linalg.norm(v) + 1e-12)

    return u, v, mu_i, mu_j, mid, mask_ij, dnorm

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

def set_border(g):
    for spine in ['top', 'bottom', 'left', 'right']:
        g.spines[spine].set_color('black')
        g.spines[spine].set_linewidth(1)

def plot_pair_scatter(
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    j: int,
    u: np.ndarray,
    v: np.ndarray,
    mid: np.ndarray,
    mu_i: np.ndarray = None,
    mu_j: np.ndarray = None,
    max_points: int = 5000,
    path: str | None = None,
):
    sns.set_theme(style="whitegrid", font_scale=3.0, rc={"xtick.bottom": True, "ytick.left": True})
    sns.set_context(rc={'patch.linewidth': 2.0})

    mask = (y == i) | (y == j)
    Xp = X[mask]
    yp = y[mask]

    if Xp.shape[0] > max_points:
        idx = np.random.choice(Xp.shape[0], size=max_points, replace=False)
        Xp = Xp[idx]
        yp = yp[idx]

    x, y_orth_energy = project_xy(Xp, u, v, mid)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(x, y_orth_energy, s=35, alpha=0.65, label="Class A")
    ax.scatter(x, y_orth_energy, s=35, alpha=0.65, label="Class B")

    ax.axvline(0.0, linewidth=2.0)
    ax.axhline(0.0, linewidth=1.0)
    set_border(ax)

    ax.set_xlabel(r"Decision-axis residual ($u_{ij}^\top (z-\mu_c)$)")
    ax.set_ylabel(r"Orthogonal magnitude ($\|(I-u_{ij}u_{ij}^\top)(z-\mu_y)\|\,$)")

    ax.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="black")
    ax.margins(x=0.05, y=0.05)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")

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

def set_border(g):
    for spine in ['top', 'bottom', 'left', 'right']:
        g.spines[spine].set_color('black')
        g.spines[spine].set_linewidth(1)

def plot_var_bars_with_errorbars(summary, epochs, k_list=(5, 20, 100), title=None, path=None, logy=True):
    # sns.set_theme(style="whitegrid", font_scale=3.0, rc={"xtick.bottom": True, "ytick.left": True})
    # sns.set_context(rc={'patch.linewidth': 2.0})
    # labels = ["Var(u)"] + [f"Top-{k} orth" for k in k_list] + ["Total orth"]
    labels = (
        [r"$\mathrm{Var}(u^\top z)$"]
        + [rf"$\sum_{{\ell=1}}^{{{k}}}\lambda_\ell^\perp$" for k in k_list]
        + [r"$\mathrm{tr}(\Sigma_\perp)$"]
    )

    metrics = ["var_u"] + [f"var_orth_top{k}" for k in k_list] + ["var_orth_total"]

    means = np.array([[summary[ep][m][0] for m in metrics] for ep in epochs], dtype=float)
    stds  = np.array([[summary[ep][m][1] for m in metrics] for ep in epochs], dtype=float)

    x = np.arange(len(epochs))
    K = len(metrics)
    width = 0.16 if K >= 5 else 0.22
    offsets = (np.arange(K) - (K - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(13, 6))
    for t in range(K):
        ax.bar(
            x + offsets[t],
            means[:, t],
            width=width,
            yerr=stds[:, t],
            capsize=4,
            label=labels[t],
        )

    set_border(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Within-class variance")
    if title:
        ax.set_title(title)

    if logy:
        ax.set_yscale("log")

    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax

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
    dm.setup('fit')
    
    # Get number of classes
    num_classes = 100
    print(f"Number of classes: {num_classes}")
    
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

    print("Generating bar plot ...")
    plot_var_bars_with_errorbars(
        summary,
        epochs,
        k_list=(1, 10, 100),
        path=f"./variance_viz/bar_pair_top10.png",
        logy=True,
    )
    # # Save results
    # df = pd.DataFrame(rows)
    # df.to_csv(args.out_csv, index=False)
    # print(f'\nWrote results to {args.out_csv}')
    # print(df.to_string())


if __name__ == '__main__':
    main()
