import sys, os, argparse, yaml, pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import random
torch.set_default_dtype(torch.float32)

from data_utils.dataloaders import get_dataset
from algorithms.factory import build_ssl_model
from eval_utils.feature_extractor import FeatureExtractor
from eval_utils.finetune_utils import FineTuneEvaluator

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures determinism
    random.seed(seed)   

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
        if encoder_type == 'vit_b':
            kwargs = {
            'patch_size': config['model']['patch_size'],
            'encoder_type': config['model']['encoder_type']

            }
    elif method == 'clip':
        kwargs = {} #

    elif method == 'mae':
        kwargs = {}

    ssl_model = build_ssl_model(
        method=config['method_type'],
        dataset=config['dataset']['name'],
        **kwargs
    )

    ssl_model.to(device)
    print(f"Loaded SSL model: {ssl_model.__class__.__name__} with encoder {encoder_type}")

    ssl_extractor = FeatureExtractor(ssl_model)
    train_features, train_labels = ssl_extractor.extract_features(train_loader)
    test_features, test_labels = ssl_extractor.extract_features(test_loader)

    embedding_layer = 0  # 0 for h, 1 for g(h)
    evaluator = FineTuneEvaluator(
        train_features = train_features[embedding_layer],
        train_labels = train_labels,
        test_features = test_features[embedding_layer],
        test_labels = test_labels,
        num_output_classes = num_output_classes,
        device = device,
        backbone = ssl_model.encoder,
        epochs = args.epochs
    )

    train_acc, test_acc = evaluator.evaluate(
        n_samples=args.n_shot,
        repeat=args.repeat
    )
    print(f"Finetune Evaluation -> Train Accuracy: {train_acc:.2%}, Test Accuracy: {test_acc:.2%}")

    # Save results to a CSV file
    row = {
        "seed": args.seed,
        "n_shot": args.n_shot,
        "train_acc": train_acc,
        "test_acc": test_acc
    }

    os.makedirs(args.output_path, exist_ok=True)
    csv_path = os.path.join(args.output_path, "results.csv")

    df = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

    print(f"Results saved to {csv_path} in directory {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Evaluation Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the SSL model checkpoint.')
    parser.add_argument('--n_shot', type=int, default=5, help='Number of shots for few-shot evaluation.')
    parser.add_argument('--repeat', type=int, default=10, help='Number of repetitions for evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--output_path', type=str, default='./finetune_results', help='Directory to save results CSV.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for finetuning.')

    args = parser.parse_args()
    main(args)
