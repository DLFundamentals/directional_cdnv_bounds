#!/usr/bin/env python3
"""
Batch CDNV Evaluation Script

Takes model checkpoints from epochs 0 to 1000, calculates CDNV and directional CDNV
at each epoch, and outputs results to a CSV file.

This script properly loads checkpoints saved by PyTorch Lightning by using the
hyperparameters stored in the checkpoint itself.

Extended to support:
- computing CDNV with either fine-grained labels or superclass labels
- loading a fine->superclass mapping from a JSON file
"""
import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import pandas as pd
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_utils.geometry import GeometricEvaluator
from training_scratch.data.mini_imagenet_datamodule import MiniImageNetCfg, MiniImageNetDataModule


def load_fine_to_super(mapping_json_path):
    """
    Load a JSON mapping of WordNet IDs to metadata and return:

    - fine_to_super: dict[int, int]   (original_index -> superclass_id)
    - num_superclasses: int
    """
    with open(mapping_json_path, 'r') as f:
        data = json.load(f)

    fine_to_super = {}
    for wnid, info in data.items():
        fine_idx = int(info['original_index'])
        super_id = int(info['superclass_id'])
        fine_to_super[fine_idx] = super_id

    num_superclasses = len(set(fine_to_super.values()))
    return fine_to_super, num_superclasses


def map_fine_to_super(labels: torch.Tensor, fine_to_super: dict) -> torch.Tensor:
    """
    Map fine-grained labels (0..N-1) to superclass labels using fine_to_super dict.

    labels: 1D LongTensor on any device
    fine_to_super: dict[int, int]
    returns: 1D LongTensor of superclass labels on same device
    """
    device = labels.device
    if labels.numel() == 0:
        return labels.clone()

    max_label = int(labels.max().item())

    # Build a lookup tensor [max_label+1] -> super_id
    lookup = []
    for i in range(max_label + 1):
        if i not in fine_to_super:
            raise KeyError(f"Fine label {i} not found in fine_to_super mapping")
        lookup.append(fine_to_super[i])

    lookup = torch.tensor(lookup, device=device, dtype=labels.dtype)
    return lookup[labels]


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

    for batch_idx, (views, y) in enumerate(loader):
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

    state_dict = ckpt.get('state_dict', {})
    model.load_state_dict(state_dict, strict=False)

    return model, cfg


def main():
    parser = argparse.ArgumentParser(description='Batch CDNV evaluation across checkpoints')
    parser.add_argument(
        '--ckpt_dir',
        default="/home/yashsalunkhe619/directional_cdnv_bounds/checkpoints/vitB_dino_mini/checkpoints",
        help='Directory containing checkpoint files'
    )
    parser.add_argument('--out_csv', default='dino_gran.csv', help='Output CSV file path')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--start', type=int, default=0, help='Start epoch')
    parser.add_argument('--end', type=int, default=1000, help='End epoch')
    parser.add_argument('--max_train_batches', type=int, default=200, help='Max batches for train features')
    parser.add_argument('--max_val_batches', type=int, default=50, help='Max batches for val features')

    # New: label-level and superclass mapping
    parser.add_argument(
        '--label_level',
        choices=['fine', 'super'],
        default='super',
        help='Use fine-grained (original) labels or superclass labels for CDNV',
    )
    parser.add_argument(
        '--superclass_mapping_json',
        type=str,
        default=None,
        help='Path to JSON file with mapping (WordNet ID -> {original_index, superclass_id, ...}) '
             'required when --label_level super',
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # Find checkpoint files
    files = find_checkpoint_files(args.ckpt_dir, start=args.start, end=args.end)
    if len(files) == 0:
        raise FileNotFoundError(
            f'No checkpoint files found in {args.ckpt_dir} for range [{args.start}, {args.end}]'
        )

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

    # Optionally load superclass mapping
    fine_to_super = None
    num_superclasses = None
    if args.label_level == 'super':
        if args.superclass_mapping_json is None:
            raise ValueError(
                "--label_level super was selected but --superclass_mapping_json was not provided"
            )
        fine_to_super, num_superclasses = load_fine_to_super(args.superclass_mapping_json)
        print(f"Loaded superclass mapping from {args.superclass_mapping_json}")
        print(f"Number of superclasses: {num_superclasses}")

    # Get number of classes depending on label level
    if args.label_level == 'fine':
        num_classes = cfg.data.get('num_classes', None) or cfg.cdnv.get('num_classes', 100)
    else:
        num_classes = num_superclasses

    print(f"Label level: {args.label_level}, num_classes: {num_classes}")

    rows = []

    for epoch, ckpt_path in files:
        print(f'Processing checkpoint: {ckpt_path} (epoch={epoch})')
        try:
            # Load model from checkpoint
            model, model_cfg = load_model_from_checkpoint(ckpt_path, device)
            model.eval()
            model.to(device)
            backbone = model.backbone

            # Extract features
            train_loader = (
                dm.probe_train_dataloader()
                if hasattr(dm, "probe_train_dataloader")
                else dm.train_dataloader()
            )
            val_loader = (
                dm.probe_test_dataloader()
                if hasattr(dm, "probe_test_dataloader")
                else dm.val_dataloader()
            )

            Xtr, Ytr_fine = extract_features(
                train_loader, backbone, device, max_batches=args.max_train_batches
            )
            Xva, Yva_fine = extract_features(
                val_loader, backbone, device, max_batches=args.max_val_batches
            )

            print(f"  Train features: {Xtr.shape}, Val features: {Xva.shape}")

            # Select / transform labels based on level
            if args.label_level == 'fine':
                Ytr = Ytr_fine
                Yva = Yva_fine
            else:
                Ytr = map_fine_to_super(Ytr_fine, fine_to_super)
                Yva = map_fine_to_super(Yva_fine, fine_to_super)

            # Sanity check: labels within range
            all_labels = torch.cat([Ytr, Yva], dim=0)
            unique_labels = torch.unique(all_labels)
            if unique_labels.min() < 0 or unique_labels.max() >= num_classes:
                raise ValueError(
                    f"Labels out of range for num_classes={num_classes}. "
                    f"Found labels {unique_labels.tolist()}"
                )

            # Compute CDNV metrics
            evaluator = GeometricEvaluator(num_classes=num_classes, device=device)
            train_cdnv = evaluator.compute_cdnv(Xtr, Ytr)
            train_dir_cdnv = evaluator.compute_directional_cdnv(Xtr, Ytr)
            val_cdnv = evaluator.compute_cdnv(Xva, Yva)
            val_dir_cdnv = evaluator.compute_directional_cdnv(Xva, Yva)

            print(
                f"  train_cdnv={train_cdnv:.6f}, "
                f"train_dir_cdnv={train_dir_cdnv:.6f}, "
                f"val_cdnv={val_cdnv:.6f}, "
                f"val_dir_cdnv={val_dir_cdnv:.6f}"
            )

            rows.append({
                'epoch': epoch,
                'label_level': args.label_level,
                'train_cdnv': train_cdnv,
                'train_dir_cdnv': train_dir_cdnv,
                'val_cdnv': val_cdnv,
                'val_dir_cdnv': val_dir_cdnv,
            })

            # Clean up to save memory
            del model, backbone, Xtr, Ytr, Xva, Yva, Ytr_fine, Yva_fine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f'ERROR processing {ckpt_path}: {e}')
            traceback.print_exc()
            rows.append({'epoch': epoch, 'label_level': args.label_level, 'error': str(e)})

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f'\nWrote results to {args.out_csv}')
    print(df.to_string())


if __name__ == '__main__':
    main()