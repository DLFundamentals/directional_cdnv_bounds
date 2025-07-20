import torch
import numpy as np
import random
from typing import List, Optional

"""
Example Usage:
evaluator = NCCCEvaluator(device='cuda')
centers, selected_classes = evaluator.compute_class_centers(
    train_features, train_labels,
    n_shot=5, repeat=5, selected_classes=[0, 1, 2, 3, 4]
)
accs = evaluator.evaluate(test_features, test_labels, centers, selected_classes)

"""

class NCCCEvaluator:
    def __init__(self, device: str = 'cuda'):
        self.device = device

    def _map_labels_to_indices(self, labels: torch.Tensor, selected_classes: List[int]):
        """
        Map original labels to 0..len(selected_classes)-1 for class subset evaluation.
        """
        label_map = {cls: i for i, cls in enumerate(selected_classes)}
        new_labels = torch.tensor([label_map[l.item()] for l in labels if l.item() in label_map],
                                  device=self.device)
        mask = torch.tensor([l.item() in label_map for l in labels], device=self.device)
        return new_labels, mask

    def compute_class_centers(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_shot: Optional[int] = None,
        repeat: int = 1,
        selected_classes: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Compute class centers from full or few-shot support set.
        If selected_classes is provided, restrict to that subset.
        """
        features = features.to(self.device)
        labels = labels.to(self.device)
        assert features.shape[0] == labels.shape[0], "Features and labels must have the same number of samples."
        
        if selected_classes is None:
            selected_classes = sorted(list(set(labels.cpu().numpy().tolist())))
        n_classes = len(selected_classes)

        mapped_labels, mask = self._map_labels_to_indices(labels, selected_classes)
        features = features[mask]

        centers_per_repeat = []

        for r in range(repeat):
            class_centers = []
            for c in range(n_classes):
                idxs = (mapped_labels == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0:
                    class_centers.append(torch.zeros(features.shape[1], device=self.device))
                    continue

                if n_shot is not None and len(idxs) >= n_shot:
                    torch.manual_seed(r)  # seed per repeat
                    idxs = idxs[torch.randperm(len(idxs))[:n_shot]]

                feats_c = features[idxs]
                center = feats_c.mean(dim=0)
                class_centers.append(center)

            centers = torch.stack(class_centers, dim=0)  # (n_classes, dim)
            centers_per_repeat.append(centers)

        return centers_per_repeat, selected_classes

    def evaluate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        centers_list: List[torch.Tensor],
        selected_classes: List[int]
    ) -> List[float]:
        """
        Evaluate accuracy using nearest-center classifier.
        Only considers selected_classes.
        """
        features = features.to(self.device)
        labels = labels.to(self.device)
        mapped_labels, mask = self._map_labels_to_indices(labels, selected_classes)
        features = features[mask]

        accs = []
        for centers in centers_list:
            dists = torch.cdist(features.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
            preds = torch.argmin(dists, dim=1)
            acc = (preds == mapped_labels).float().mean().item()
            accs.append(acc)

        return accs