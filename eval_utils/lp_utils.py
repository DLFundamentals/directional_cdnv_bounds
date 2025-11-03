import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Optional, Tuple

"""
Example Usage:

evaluator = LinearProbeEvaluator(
    train_features=h_train,
    train_labels=y_train,
    test_features=h_test,
    test_labels=y_test,
    output_classes=5,
    label_subset=[0, 1, 2, 3, 4],
    device='cuda',
)

train_acc, test_acc = evaluator.evaluate(n_samples=5, repeat=10)
print(f"Train: {train_acc:.2%}, Test: {test_acc:.2%}")

"""

class LinearProbeEvaluator:
    """
    Train and evaluate a linear classifier on frozen SSL features.
    
    Features should already be extracted from the encoder (either h or g(h)).
    Supports full or few-shot training, class-subset selection, and repeat evaluation.
    """

    def __init__(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        num_output_classes: int,
        device: str = "cuda",
        lr: float = 3e-4,
        epochs: int = 100,
        selected_classes: Optional[List[int]] = None,
    ):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.num_output_classes = num_output_classes

        self.train_features = train_features.to(device)
        self.train_labels = train_labels.to(device)
        self.test_features = test_features.to(device)
        self.test_labels = test_labels.to(device)

        self.selected_classes = selected_classes or sorted(list(set(train_labels.cpu().numpy().tolist())))
        self.label_map = {label: idx for idx, label in enumerate(self.selected_classes)}

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _map_labels_and_filter(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map class subset labels to 0..N-1 and filter features accordingly.
        """
        label_mask = torch.tensor([l.item() in self.label_map for l in labels], device=self.device)
        filtered_feats = features[label_mask]
        mapped_labels = torch.tensor(
            [self.label_map[l.item()] for l in labels[label_mask]], device=self.device
        )
        return filtered_feats, mapped_labels

    def _sample_fewshot(
        self, features: torch.Tensor, labels: torch.Tensor, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample `n_samples` per class from filtered features.
        """
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels.cpu().tolist()):
            class_to_indices[label].append(idx)

        selected_indices = []
        for c in self.label_map.values():
            indices = class_to_indices.get(c, [])
            if len(indices) < n_samples:
                raise ValueError(f"Class {c} has only {len(indices)} samples")
            selected = random.sample(indices, n_samples)
            selected_indices.extend(selected)

        selected_indices = torch.tensor(selected_indices, device=self.device)
        return features[selected_indices], labels[selected_indices]

    def _train_probe(
        self, features: torch.Tensor, labels: torch.Tensor, input_dim: int
    ) -> torch.nn.Module:
        """
        Train linear classifier on frozen features.
        """
        probe = torch.nn.Linear(input_dim, self.num_output_classes, bias=False).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            probe.train()
            logits = probe(features)
            loss = self.loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return probe

    @torch.no_grad()
    def _evaluate_probe(
        self, probe: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor
    ) -> float:
        probe.eval()
        preds = torch.argmax(probe(features), dim=1)
        acc = (preds == labels).float().mean().item()
        return acc

    def evaluate(
        self,
        n_samples: Optional[int] = None,
        repeat: int = 5,
    ) -> Tuple[float, float]:
        """
        Train and evaluate a linear probe using frozen features.

        Args:
            n_samples: Number of samples per class for few-shot evaluation. If None, use full dataset.
            repeat: Number of repeated runs (with new few-shot samples each time).

        Returns:
            Average train and test accuracy.
        """
        train_feats, train_labels = self._map_labels_and_filter(
            self.train_features, self.train_labels
        )
        test_feats, test_labels = self._map_labels_and_filter(
            self.test_features, self.test_labels
        )

        input_dim = train_feats.shape[1]

        train_accs, test_accs = [], []

        for r in range(repeat):
            if n_samples is not None:
                fewshot_feats, fewshot_labels = self._sample_fewshot(train_feats, train_labels, n_samples)
            else:
                fewshot_feats, fewshot_labels = train_feats, train_labels

            probe = self._train_probe(fewshot_feats, fewshot_labels, input_dim)

            train_acc = self._evaluate_probe(probe, train_feats, train_labels)
            test_acc = self._evaluate_probe(probe, test_feats, test_labels)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

        return float(np.mean(train_accs)), float(np.mean(test_accs))
