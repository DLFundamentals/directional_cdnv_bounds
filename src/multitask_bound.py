"""
How to run?
python multi_labeling.py --config configs/cifar10_resnet50.yaml \
    --ckpt_path checkpoints/cifar10_parallel \
    --output_path results/cifar10 \
    --supervision dcl # dcl/nscl/scl/ce
"""

import torch
torch.set_default_dtype(torch.float32)
from torch.utils.data import DataLoader
import torchvision.models as models
import sys, os, argparse, yaml, pandas as pd

# Append the parent directory for utility modules.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and models.
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from models.simclr import SimCLR, SimCLRWithClassificationHead

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

set_seed(42)

def load_snapshot(snapshot_path, model, device):
    snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
    state_dict = snapshot['MODEL_STATE']
    epochs_trained = snapshot['EPOCHS_RUN']
    print(f"Loaded model from epoch {epochs_trained}")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def pretty_combo(a, b):
    def pretty_label(s):
        s = s.replace("_label", "").replace("_", " ")
        return s.title()
    return f"{pretty_label(a)} vs. {pretty_label(b)}"

def compute_label_geometry(features, labels, label_key, eps=1e-12):
    """
    Returns:
        {
            "K": int,
            "pairs": [
                {
                    "class1": int,
                    "class2": int,
                    "u": tensor[D],
                    "d": float,
                    "V": float
                }
            ]
        }
    """
    device = features.device
    y = torch.from_numpy(labels[label_key]).to(device)

    classes = torch.unique(y)
    K = len(classes)

    means = torch.stack([features[y == c].mean(dim=0) for c in classes])

    ii, jj = torch.triu_indices(K, K, offset=1, device=device)

    pairs = []

    for idx in range(len(ii)):
        i = ii[idx]
        j = jj[idx]

        ci = classes[i].item()
        cj = classes[j].item()

        mu_i = means[i]
        mu_j = means[j]

        diff = mu_i - mu_j
        d = diff.norm().clamp_min(eps)
        u = diff / d

        mask = (y == classes[i]) | (y == classes[j])
        X_pair = features[mask]
        y_pair = y[mask]

        centered = torch.zeros_like(X_pair)
        centered[y_pair == classes[i]] = X_pair[y_pair == classes[i]] - mu_i
        centered[y_pair == classes[j]] = X_pair[y_pair == classes[j]] - mu_j

        proj = centered @ u
        proj_var = (proj ** 2).mean()
        V = proj_var / (d ** 2)

        pairs.append({
            "class1": ci,
            "class2": cj,
            "u": u,
            "d": float(d),
            "V": float(V),
        })

    return {"K": K, "pairs": pairs}

ALL_LABEL_COMBINATIONS = [
    ("color", "shape"), ("color", "style"), ("color", "size_label"),
    ("shape", "style"), ("shape", "size_label"), ("style", "size_label"),
]

ALL_LABEL_KEYS = ['color', 'shape', 'size_label', 'style']

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="General Evluation Script")
    parser.add_argument('--config', '-c', required=True, help='path to yaml config file')
    parser.add_argument('--ckpt_path', '-ckpt', required=True, default=None,
                        help='path to model checkpoints')
    # parser.add_argument('--output_path', '-out', required=True, default=None,
    #                     help='path to save logs')
    parser.add_argument('--supervision', default='dcl', type=str, choices=['dcl', 'nscl', 'scl', 'ce'],
                        help='which model to evaluate')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # load config parameters required for evaluation
    experiment_name = config['experiment_name']
    method_type = config['method_type']

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']
    num_output_classes = config['dataset']['num_output_classes']
    if dataset_name == 'synthetic':
        label_key = config['dataset'].get('label_key', None)

    batch_size = config['training']['batch_size']
    augment_both = config['training']['augment_both']

    encoder_type = config['model']['encoder_type']
    width_multiplier = config['model']['width_multiplier']
    projection_dim = config['model']['projection_dim']
    hidden_dim = config['model']['hidden_dim']

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get dataset
    augment_both = False # override for evaluation
    train_dataset, _, test_dataset, _, train_labels, test_labels = get_dataset(dataset_name=dataset_name, 
                                                                    dataset_path=dataset_path,
                                                                    augment_both_views=augment_both,
                                                                    batch_size=batch_size, test=True,
                                                                    label_key=label_key)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    # breakpoint()
    # load model
    if encoder_type == 'resnet18':
        encoder = models.resnet18(weights=None)
        PATCH_SIZE = None
        IMAGE_SIZE = None
        HIDDEN_DIM = None
        MLP_DIM = None
        STRIDE = None
    elif encoder_type == 'resnet50':
        encoder = models.resnet50(pretrained=False)
    elif encoder_type == 'vit_b':
        image_size = 224 if 'imagenet' in dataset_name else 32
        # encoder = models.vit_b_16(weights=None, image_size=image_size)
        encoder = models.VisionTransformer(
            patch_size=16 if 'imagenet' in dataset_name else 4,
            image_size=224 if 'imagenet' in dataset_name else 32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768 if 'imagenet' in dataset_name else 384,
            mlp_dim=3072 if 'imagenet' in dataset_name else 1536,
        )
    else:
        raise NotImplementedError(f"{encoder_type} not implemented")
    
    if method_type == 'simclr':
        model = SimCLR(model=encoder,
                        dataset=dataset_name,
                        width_multiplier=width_multiplier,
                        hidden_dim=hidden_dim,
                        projection_dim=projection_dim,
                        # hyperparams for ViT
                        image_size = 224 if 'imagenet' in dataset_name else 32,
                        patch_size = 16 if 'imagenet' in dataset_name else 4,
                        stride = 16 if 'imagenet' in dataset_name else 2,
                        token_hidden_dim = 768 if 'imagenet' in dataset_name else 384,
                        mlp_dim = 3072 if 'imagenet' in dataset_name else 1536,
                        )
    else:
        raise NotImplementedError(f"{method_type} not implemented")

    if args.supervision == 'ce':
        model = SimCLRWithClassificationHead(
            simclr_model=model,
            num_classes=num_output_classes
        )
    
    # load model checkpoint
    checkpoints_dir = f'{args.ckpt_path}/{args.supervision}'
    print(f"Loading checkpoints from {checkpoints_dir}")
    checkpoint_files = os.listdir(checkpoints_dir)
    sorted_checkpoints = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    OUT_DIR = "/home/luthra/directional_cdnv_bounds/multi_labelings/results/"
    ALL_PAIR_METRICS = os.path.join(OUT_DIR, "all_pair_metrics.csv")
    ALL_PAIR_COS = os.path.join(OUT_DIR, "all_pair_cos.csv")
    
    epochs = [0, 10, 50, 100, 500, 1000]
    all_metric_rows = []
    lhs_rows = []

    for ssl_ckpt in sorted_checkpoints:
        epoch = int(ssl_ckpt.split('_')[-1].split('.')[0])
        if epoch not in epochs:
            print(f"Epoch {epoch} already evaluated. Skipping.")
            continue
        print(f'\nEvaluating Epoch {epoch}')
        snapshot_path = f'{args.ckpt_path}/{args.supervision}/snapshot_{epoch}.pth'
        model = load_snapshot(snapshot_path, model, device)
        model.eval()
        model = freeze_model(model)
        print("Models frozen for feature extraction")
        # --- Feature Extraction ---
        emb_layer = 0 # 0 for h and 1 for g(h)
        extractor = FeatureExtractor(model)
        train_features, _ = extractor.extract_features(train_loader)
        # test_features, _ = extractor.extract_features(test_loader)
        
        # --- Evaluation ---
        label_geometries = {}

        for label_key in ALL_LABEL_KEYS:
            geom = compute_label_geometry(
                train_features[emb_layer],
                train_labels,
                label_key
            )
            label_geometries[label_key] = geom

            for p in geom["pairs"]:
                all_metric_rows.append({
                    "epoch": epoch,
                    "label_key": label_key,
                    "class1": p["class1"],
                    "class2": p["class2"],
                    "d": p["d"],
                    "K": geom["K"],
                    "V": p["V"],
        })

        for (key1, key2) in ALL_LABEL_COMBINATIONS:

            geom1 = label_geometries[key1]
            geom2 = label_geometries[key2]

            for p1 in geom1["pairs"]:
                for p2 in geom2["pairs"]:

                    cos_val = torch.abs(
                        torch.dot(p1["u"], p2["u"])
                    ).item()

                    lhs_rows.append({
                        "epoch": epoch,
                        "label_key1": key1,
                        "key1_class1": p1["class1"],
                        "key1_class2": p1["class2"],
                        "label_key2": key2,
                        "key2_class1": p2["class1"],
                        "key2_class2": p2["class2"],
                        "mod_cos_sim": cos_val,
                    })
    
    # save results
    all_metrics_df = pd.DataFrame(all_metric_rows)
    all_metrics_df.to_csv(ALL_PAIR_METRICS, index=False)
    lhs_df = pd.DataFrame(lhs_rows)
    lhs_df.to_csv(ALL_PAIR_COS, index=False)