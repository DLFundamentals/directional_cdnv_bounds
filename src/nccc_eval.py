import sys, os, argparse, yaml, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_default_dtype(torch.float32)

from encoders.get_encoders import build_ssl_encoder
from data_utils.dataloaders import get_dataset
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.nccc_utils import NCCCEvaluator
from eval_utils.nccc_utils import NCCCEvaluator

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism

def main(args):
    set_seed(args.seed)
    # set device
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # build dataset
    _, train_loader, _, test_loader, train_labels, test_labels = get_dataset(
        dataset_name=config['dataset']['name'],
        dataset_path=config['dataset']['path'],
        augment_both_views=config['linear']['augment_both'],
        batch_size=config['linear']['batch_size'],
        test=True,
    )
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")

    # build encoder
    encoder_type = config['model']['encoder_type']
    if encoder_type == 'resnet50':
        kwargs = {
            'width_multiplier': config['model'].get('width_multiplier', 1),
            'hidden_dim': config['model'].get('hidden_dim', 2048),
            'projection_dim': config['model'].get('projection_dim', 128),
        }
    elif encoder_type == 'vit_b':
        kwargs = {
            'use_old': False
        }
    elif encoder_type == 'ijepa':
        kwargs = {} # @HeisenbergsCat03 do we need any arguments here?

    elif encoder_type == 'clip':
        kwargs = {}

    ssl_model = build_ssl_encoder(
        method=config['method_type'],
        encoder_type=config['model']['encoder_type'],
        dataset=config['dataset']['name'],
        checkpoint=args.ckpt_path,
        device=device,
        **kwargs
    )
    freeze_model(ssl_model)
    print(f"Loaded SSL model: {ssl_model.__class__.__name__} with encoder {encoder_type}")

    # extract features
    ssl_extractor = FeatureExtractor(ssl_model)
    train_features, train_labels = ssl_extractor.extract_features(train_loader)
    test_features, test_labels = ssl_extractor.extract_features(test_loader)

    # initialize evaluator
    # TODO: make n_shot, repeat, selected_classes configurable
    embedding_layer = 0 # 0 for h, 1 for g(h)
    evaluator = NCCCEvaluator(device=device)
    centers, selected_classes = evaluator.compute_class_centers(
        test_features[embedding_layer], test_labels,
        n_shot=100,
        repeat=1,
        selected_classes=None
    )
    # make sure to use above selected classes while evaluating
    accs = evaluator.evaluate(
        test_features[embedding_layer], test_labels, centers, selected_classes
    )
    print(f"Evaluation accuracies: {accs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCCC Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--ckpt_path', type=str, help='Path to the SSL model checkpoint')
    parser.add_argument('--output_path', type=str, default='logs/nccc', help='Path to save evaluation results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--repeat', type=int, default=1, help='Number of repeats for evaluation')
    parser.add_argument('--n_shot', type=int, default=100, help='Number of shots for few-shot evaluation')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    main(args)