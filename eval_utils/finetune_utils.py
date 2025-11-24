import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Optional, Tuple

class FineTuneEvaluator:
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
            backbone: torch.nn.Module = None,
            selected_classes: Optional[List[int]] = None,
        ):
            self.train_features = train_features
            self.train_labels = train_labels
            self.test_features = test_features
            self.test_labels = test_labels
            self.num_output_classes = num_output_classes
            self.device = device
            self.lr = lr
            self.epochs = epochs
            self.selected_classes = selected_classes
            self.loss_fn = torch.nn.CrossEntropyLoss()

            self.backbone = backbone.to(device) if backbone else None
            self.label_map = {label: idx for idx, label in enumerate(self.selected_classes)} if selected_classes else None  

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

    def finetune_model(self, train_features: torch.Tensor, train_labels: torch.Tensor) -> torch.nn.Module:
        """
        Finetune the original model as well as the linear classifier on top of it.
        The training loop should have the backbone model not frozen and its parameters updated.
        """
        if self.backbone is None:
            raise ValueError("Backbone model must be provided for finetuning.")

        self.backbone.train()
        input_dim = train_features.shape[1]
        classifier = torch.nn.Linear(input_dim, len(self.selected_classes)).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(classifier.parameters()), lr=self.lr
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = classifier(self.backbone(train_features))
            loss = self.loss_fn(outputs, train_labels)
            loss.backward()
            optimizer.step()

        return classifier
    
    @torch.no_grad()
    def evaluate_model(
        self, model: torch.nn.Module, test_features: torch.Tensor, test_labels: torch.Tensor
    ) -> float:
        """
        Evaluate the finetuned model on the test set.
        """
        model.eval()
        outputs = model(self.backbone(test_features))
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_labels).sum().item()
        total = test_labels.size(0)
        accuracy = correct / total
        return accuracy
    
    def evaluate(
        self,
        n_samples: Optional[int] = None,
        repeat: int = 5,
    ) -> Tuple[float, float]:
        """
        Train and evaluate a finetuned model using frozen features.

        Args:
            n_samples (Optional[int]): Number of samples per class for few-shot learning. If None, use all data.
            repeat (int): Number of times to repeat the evaluation for averaging.

        Returns:
            Tuple[float, float]: Average train and test accuracy.
        """
        train_feats, train_labels = self._map_labels_and_filter(
            self.train_features, self.train_labels
        )
        test_feats, test_labels = self._map_labels_and_filter(
            self.test_features, self.test_labels
        )

        if n_samples is not None:
            train_feats, train_labels = self._sample_fewshot(
                train_feats, train_labels, n_samples
            )

        train_accs, test_accs = [], []

        for _ in range(repeat):
            finetuned_model = self.finetune_model(train_feats, train_labels)
            train_acc = self.evaluate_model(finetuned_model, train_feats, train_labels)
            test_acc = self.evaluate_model(finetuned_model, test_feats, test_labels)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

        avg_train_acc = np.mean(train_accs)
        avg_test_acc = np.mean(test_accs)

        return float(avg_train_acc), float(avg_test_acc)


        
