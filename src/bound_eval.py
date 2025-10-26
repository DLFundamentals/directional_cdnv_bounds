import sys, os, argparse, yaml, json, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
torch.set_default_dtype(torch.float32)

from data_utils.dataloaders import get_dataset
from algorithms.factory import build_ssl_model
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.geometry import GeometricEvaluator

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

    import json

def save_pairwise_metrics(metrics, path):
    # convert (i, j) -> "i_j"
    metrics_str_keys = {f"{i}_{j}": v for (i, j), v in metrics.items()}
    with open(path, "w") as f:
        json.dump(metrics_str_keys, f, indent=2)

def main(args):
    set_seed(args.seed)
    # set device
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # build dataset
    num_output_classes = config['dataset']['num_output_classes']
    classes_groups = random.sample(range(num_output_classes),2)
    classes_groups = None
    _, train_loader, _, test_loader, train_labels, test_labels = get_dataset(
        method = config['method_type'],
        dataset_name=config['dataset']['name'],
        dataset_path=config['dataset']['path'],
        augment_both_views=config['linear']['augment_both'],
        batch_size=config['linear']['batch_size'],
        test=True,
        classes = classes_groups
    )
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    # set kwargs
    method = config['method_type']
    encoder_type = config['model']['encoder_type']
    if method == 'simclr':
        if encoder_type == 'resnet50':
            kwargs = {
                'width_multiplier': config['model'].get('width_multiplier', 1),
                'hidden_dim': config['model'].get('hidden_dim', 2048),
                'projection_dim': config['model'].get('projection_dim', 128),
                'ckpt_path': args.ckpt_path,
            }
        elif encoder_type == 'vit_b':
            kwargs = {
                # TODO
            }
    elif method == 'ijepa':
        kwargs = {
            'patch_size': config['model']['patch_size'],
            'encoder_type': config['model']['encoder_type']
        }

    elif method == 'clip':
        kwargs = {} # TODO

    ssl_model = build_ssl_model(
        method=config['method_type'],
        dataset=config['dataset']['name'],
        **kwargs
    )
    ssl_model.to(device)
    freeze_model(ssl_model)
    print(f"Loaded SSL model: {ssl_model.__class__.__name__} with encoder {encoder_type}")

    # extract features
    ssl_extractor = FeatureExtractor(ssl_model)
    train_features, train_labels = ssl_extractor.extract_features(train_loader)
    test_features, test_labels = ssl_extractor.extract_features(test_loader)

    # compute pairwise metrics for bound evaluation
    emb_layer = 0
    geom_evaluator = GeometricEvaluator(num_output_classes)
    train_pairwise_metrics = geom_evaluator.compute_pairwise_metrics(train_features[emb_layer], train_labels)
    test_pairwise_metrics = geom_evaluator.compute_pairwise_metrics(test_features[emb_layer], test_labels)

    # save results
    train_log_file = f"{args.output_path}/train_pairwise_metrics.json"
    save_pairwise_metrics(train_pairwise_metrics, train_log_file)
    test_log_file = f"{args.output_path}/test_pairwise_metrics.json"
    save_pairwise_metrics(test_pairwise_metrics, test_log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCCC Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--ckpt_path', type=str, help='Path to the SSL model checkpoint')
    parser.add_argument('--output_path', type=str, default='logs/simclr/geometry', help='Path to save evaluation results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    main(args)