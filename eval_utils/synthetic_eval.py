import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)
from torch.utils.data import DataLoader
import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd
import json
import numpy as np
from omegaconf import OmegaConf

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_scratch'))

# Import utility functions and models.
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.geometry import GeometricEvaluator
from models.dino import LightlyDINO
from models.vicreg import LightlyVICReg
from synthetic_utils import LABEL_MAPS

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42)

def _clean_lightning_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            cleaned[k[6:]] = v
        else:
            cleaned[k] = v
    return cleaned


def load_model_from_lightning_checkpoint(ckpt_path: str, method: str, device: str):
    """Load a LightningModule checkpoint and return (model, cfg)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', None)
    cfg = OmegaConf.create(hparams) if hparams is not None else None

    if method == 'dino':
        if cfg is None:
            cfg = OmegaConf.create({
                'method': {'name': 'dino'},
                'model': {
                    'ibot_separate_head': False,
                    'drop_path_rate': 0.1,
                    'mask_patch_size': 8,
                    'mask_ratio': 0.6,
                },
                'data': {'name': 'synthetic_shapes', 'img_size': 224, 'batch_size': 256},
                'trainer': {'max_epochs': 1000},
            })
        model = LightlyDINO(cfg)
    elif method == 'vicreg':
        if cfg is None:
            raise ValueError(f"Checkpoint {ckpt_path} missing hyper_parameters; cannot init VICReg")
        model = LightlyVICReg(cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    state_dict = ckpt.get('state_dict', None)
    if state_dict is None and isinstance(ckpt, dict):
        state_dict = ckpt
    model.load_state_dict(_clean_lightning_state_dict(state_dict), strict=False)
    model = model.to(device)
    model.eval()
    return model, cfg


def extract_backbone_features(backbone, images: torch.Tensor) -> torch.Tensor:
    """Return features [B, D] for ViT/ResNet-style backbones."""
    vit = getattr(backbone, 'vit', None)
    if vit is not None:
        out = vit.forward_features(images)
        if isinstance(out, torch.Tensor) and out.dim() == 3:
            return out[:, 0]
        return out

    if hasattr(backbone, 'forward_features'):
        out = backbone.forward_features(images)
    else:
        out = backbone(images)

    if isinstance(out, torch.Tensor) and out.dim() == 3:
        out = out[:, 0]
    elif isinstance(out, torch.Tensor) and out.dim() > 2:
        out = torch.flatten(out, 1)
    return out


def extract_features(model, data_loader, device: str) -> torch.Tensor:
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        raise RuntimeError('Model has no .backbone for feature extraction')

    model.eval()
    all_features = []
    with torch.no_grad():
        for images in data_loader:
            images = images.to(device)
            feats = extract_backbone_features(backbone, images)
            feats = F.normalize(feats, dim=1)
            all_features.append(feats.cpu())
    return torch.cat(all_features, dim=0)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def cosine2_samples_between_labelings(
    features: torch.Tensor,   # [N, D]
    labels: dict,             # {key: LongTensor[N]}
    label_keys: tuple[str, str],
    eps: float = 1e-12,
    max_pairs_per_labeling: int | None = 500,  # subsample pairwise directions per labeling
    max_cross_samples: int | None = 5000,      # subsample cross-pairs cos^2 values
    seed: int = 0,
):
    """
    Returns:
      cos2_samples: 1D torch tensor of sampled cos^2 values between all directions from labeling A and labeling B
      mean_cos2: scalar torch tensor
    """
    torch.manual_seed(seed)
    device = features.device
    k1, k2 = label_keys
    y1 = torch.from_numpy(labels[k1]).to(device)
    y2 = torch.from_numpy(labels[k2]).to(device)

    def class_means(y):
        classes = torch.unique(y)
        means = []
        for c in classes:
            means.append(features[y == c].mean(dim=0))
        return torch.stack(means, dim=0)  # [C, D]

    def pair_dirs(means):
        C = means.size(0)
        ii, jj = torch.triu_indices(C, C, offset=1, device=device)
        diffs = means[ii] - means[jj]
        norms = diffs.norm(dim=1, keepdim=True).clamp_min(eps)
        dirs = diffs / norms  # [P, D]
        return dirs

    mu1 = class_means(y1)
    mu2 = class_means(y2)

    d1 = pair_dirs(mu1)
    d2 = pair_dirs(mu2)

    # subsample directions within each labeling if requested
    if max_pairs_per_labeling is not None:
        if d1.size(0) > max_pairs_per_labeling:
            idx = torch.randperm(d1.size(0), device=device)[:max_pairs_per_labeling]
            d1 = d1[idx]
        if d2.size(0) > max_pairs_per_labeling:
            idx = torch.randperm(d2.size(0), device=device)[:max_pairs_per_labeling]
            d2 = d2[idx]

    # full cross cosine matrix (dirs are normalized => cosine = dot)
    cos = d1 @ d2.T                      # [P1, P2]
    cos2 = cos.pow(2).flatten()          # [P1*P2]

    # subsample cross cos^2 values for storage
    if max_cross_samples is not None and cos2.numel() > max_cross_samples:
        idx = torch.randperm(cos2.numel(), device=device)[:max_cross_samples]
        cos2_s = cos2[idx]
    else:
        cos2_s = cos2

    return cos2_s.detach().cpu(), cos2.mean().detach().cpu()

def pretty_combo(a, b):
    def pretty_label(s):
        s = s.replace("_label", "").replace("_", " ")
        return s.title()
    return f"{pretty_label(a)} vs. {pretty_label(b)}"

ALL_LABEL_COMBINATIONS = [
    ("color", "shape"), ("color", "style"), ("color", "size_label"),
    ("shape", "style"), ("shape", "size_label"), ("style", "size_label"),
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Synthetic Shapes Evaluation Script")
    parser.add_argument('--method', '-m', type=str, default='dino', choices=['dino', 'vicreg', 'simclr'],
                        help='which method to evaluate')
    parser.add_argument('--ckpt_dir', '-ckpt_dir', type=str, default=None,
                        help='directory containing checkpoint files (use with --epochs)')
    parser.add_argument('--ckpt_path', '-ckpt', type=str, default=None,
                        help='path to single checkpoint file')
    parser.add_argument('--output_dir', '-out', type=str, default='./synthetic_eval_results',
                        help='directory to save evaluation results')
    parser.add_argument('--data_root', type=str, default='./synthetic_shapes',
                        help='root directory of synthetic shapes dataset')
    parser.add_argument('--epochs', type=int, nargs='+', default=None,
                        help='specific epochs to evaluate (e.g., --epochs 0 10 50 100 500 1000)')
    parser.add_argument('--skip_epoch0', action='store_true',
                        help='skip evaluating epoch 0 even if epoch_0000.ckpt exists')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load synthetic shapes metadata for labels
    metadata_path = os.path.join(args.data_root, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Build train/val split (consistent with training)
    n = len(metadata)
    val_fraction = 0.05
    val_size = max(1, int(n * val_fraction))
    seed = 123
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx = set(indices[:val_size])
    train_idx = [i for i in range(n) if i not in val_idx]

    train_recs = [metadata[i] for i in train_idx]
    test_recs = [metadata[i] for i in sorted(val_idx)]
    
    # Labels are taken directly from metadata.json via LABEL_MAPS
    labels_train = {
        k: np.array([LABEL_MAPS[k][r[k]] for r in train_recs], dtype=np.int64)
        for k in LABEL_MAPS
    }
    labels_test = {
        k: np.array([LABEL_MAPS[k][r[k]] for r in test_recs], dtype=np.int64)
        for k in LABEL_MAPS
    }

    print(f"Loaded {len(train_recs)} training samples")
    print(f"Loaded {len(test_recs)} test samples")
    print(f"Available label keys: {list(LABEL_MAPS.keys())}")

    # Load dataset for feature extraction (images come from metadata.json records)
    from torchvision import transforms

    from PIL import Image
    from torch.utils.data import Dataset

    class ImagesFromRecordsDataset(Dataset):
        def __init__(self, root_dir, records, transform):
            self.root_dir = root_dir
            self.records = records
            self.transform = transform

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            img_path = os.path.join(self.root_dir, r['file'])
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
    
    if args.method in ('dino', 'vicreg'):
        print(f"\n=== Evaluating {args.method.upper()} model ===")
        
        # Determine checkpoint paths
        if args.epochs and args.ckpt_dir:
            # Multiple epochs from directory
            checkpoint_paths = []
            for epoch in args.epochs:
                if args.skip_epoch0 and epoch == 0:
                    continue
                ckpt_path = os.path.join(args.ckpt_dir, f'epoch_{epoch:04d}.ckpt')
                if os.path.exists(ckpt_path):
                    checkpoint_paths.append((epoch, ckpt_path))
                else:
                    print(f"Warning: checkpoint not found at {ckpt_path}")
        elif args.ckpt_path:
            # Single checkpoint
            checkpoint_paths = [(None, args.ckpt_path)]
        else:
            raise ValueError("Either --ckpt_path or (--ckpt_dir and --epochs) must be provided")
        
        print(f"Evaluating {len(checkpoint_paths)} checkpoint(s)")
        
        # Extract features
        batch_size = 256
        img_size = 224

        transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        train_ds = ImagesFromRecordsDataset(
            root_dir=args.data_root,
            records=train_recs,
            transform=transform,
        )
        test_ds = ImagesFromRecordsDataset(
            root_dir=args.data_root,
            records=test_recs,
            transform=transform,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # Compute evaluation metrics for all checkpoints
        D = None
        baseline = None
        mean_rows = []
        sample_rows = []
        
        for epoch, ckpt_path in checkpoint_paths:
            print(f"\n  Processing checkpoint: {ckpt_path}")
            if epoch is not None:
                print(f"  Epoch: {epoch}")
            
            # Load checkpoint
            model, _cfg = load_model_from_lightning_checkpoint(ckpt_path, args.method, device)
            model = freeze_model(model)
            
            # Extract features
            train_features = extract_features(model, train_loader, device)
            print(f"  Extracted features shape: {train_features.shape}")
            
            if D is None:
                D = train_features.shape[1]
                baseline = 1.0 / D
            
            # Compute metrics
            for (a, b) in ALL_LABEL_COMBINATIONS:
                cos2_samples, mean_cos2 = cosine2_samples_between_labelings(
                    train_features,
                    labels_train,
                    (a, b),
                    max_pairs_per_labeling=500,
                    max_cross_samples=5000,
                    seed=epoch if epoch is not None else 42,
                )
                
                combo = pretty_combo(a, b)
                row_dict = {
                    "combo": combo,
                    "mean_cos2": float(mean_cos2),
                    "baseline_1_over_d": baseline,
                    "dimension": D,
                }
                if epoch is not None:
                    row_dict["epoch"] = epoch
                
                mean_rows.append(row_dict)
                
                sample_dict = {
                    "combo": combo,
                    "cos2_samples": cos2_samples.numpy(),
                    "baseline_1_over_d": baseline,
                }
                if epoch is not None:
                    sample_dict["epoch"] = epoch
                
                sample_rows.append(sample_dict)
                
                print(f"  {combo}: mean_cos2={mean_cos2.item():.6f}, baseline={baseline:.6f}")
        
        # Save results
        csv_mean = os.path.join(args.output_dir, f"{args.method}_cos2_mean.csv")
        pd.DataFrame(mean_rows).to_csv(csv_mean, index=False)
        print(f"\nSaved mean results to {csv_mean}")
        
        # Save sample-level results
        csv_samples = os.path.join(args.output_dir, f"{args.method}_cos2_samples.csv")
        sample_data = []
        for row_dict in sample_rows:
            combo = row_dict['combo']
            epoch = row_dict.get('epoch')
            for cos2_val in row_dict['cos2_samples']:
                sample_entry = {'combo': combo, 'cos2': float(cos2_val)}
                if epoch is not None:
                    sample_entry['epoch'] = epoch
                sample_data.append(sample_entry)
        pd.DataFrame(sample_data).to_csv(csv_samples, index=False)
        print(f"Saved sample results to {csv_samples}")
        
    elif args.method == 'simclr':
        raise NotImplementedError("SimCLR evaluation not yet implemented in this version")
    else:
        raise ValueError(f"Unknown method: {args.method}")