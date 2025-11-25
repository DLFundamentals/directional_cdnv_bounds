import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Optional, Tuple
from torch.utils.data import TensorDataset, DataLoader, Subset
import copy

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
            train_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
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
            self.train_loader = train_loader
            self.test_loader = test_loader

            self.backbone = backbone.to(device) if backbone else None
            if self.selected_classes is None:
                try:
                    unique = sorted(list(set(train_labels.cpu().numpy().tolist())))
                except Exception:
                    unique = []
                self.selected_classes = unique

            self.label_map = {label: idx for idx, label in enumerate(self.selected_classes)}

    def _map_labels_and_filter(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map class subset labels to 0..N-1 and filter features accordingly.
        """
        # Ensure labels are on the same device as features to avoid indexing errors
        target_device = features.device
        labels = labels.to(target_device)

        # Build boolean mask on the features device
        label_list = labels.cpu().tolist()
        mask_list = [int(l) in self.label_map for l in label_list]
        label_mask = torch.tensor(mask_list, dtype=torch.bool, device=target_device)

        filtered_feats = features[label_mask]

        # Map filtered labels to 0..N-1 and place on target_device
        filtered_labels = labels[label_mask].cpu().tolist()
        mapped = [self.label_map[int(l)] for l in filtered_labels]
        mapped_labels = torch.tensor(mapped, device=target_device)

        return filtered_feats, mapped_labels
    
    def _sample_fewshot(
        self, features: torch.Tensor, labels: torch.Tensor, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Randomly sample `n_samples` per class from filtered features.
        Returns the indices of selected samples.
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

        idx_device = features.device
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=idx_device)
        return features[selected_indices], labels[selected_indices], selected_indices.cpu().numpy()

    def _create_fewshot_loader(self, loader: DataLoader, selected_indices: np.ndarray, batch_size: int = 64):
        """
        Create a DataLoader with only the selected few-shot samples.
        """
        # Create a subset of the dataset
        subset = Subset(loader.dataset, selected_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)

    def _extract_features(self, backbone_output):
        """
        Helper function to extract tensor features from various backbone output types.
        """
        if isinstance(backbone_output, torch.Tensor):
            return backbone_output
        elif hasattr(backbone_output, 'last_hidden_state'):
            # For transformers like MAE, ViT, etc.
            return backbone_output.last_hidden_state[:, 0]  # Use CLS token
        elif isinstance(backbone_output, dict):
            # If it returns a dict, try to get the main output
            feat = backbone_output.get('last_hidden_state', backbone_output.get('pooler_output'))
            if feat is not None and len(feat.shape) > 2:
                return feat[:, 0]
            return feat
        else:
            raise TypeError(f"Unexpected backbone output type: {type(backbone_output)}")

    def finetune_model(self, train_dataloader: DataLoader):
        """
        Finetune the backbone end-to-end with a linear classifier on top.
        
        Args:
            train_dataloader: DataLoader yielding (images, labels) pairs
        
        Returns:
            Tuple of (backbone, classifier)
        """
        if self.backbone is None:
            raise ValueError("Backbone must be provided")

        if train_dataloader is None:
            raise ValueError("To fine-tune the backbone you must provide train_dataloader with images")

        # Set backbone to training mode
        self.backbone.train()
        
        # Get feature dimension by doing a forward pass
        with torch.no_grad():
            sample_batch = next(iter(train_dataloader))
            
            # Handle different batch formats
            if isinstance(sample_batch, (list, tuple)):
                # Find images and labels by checking tensor shapes
                for item in sample_batch:
                    if isinstance(item, torch.Tensor):
                        # Images typically have shape [B, C, H, W] with C=3 and H,W > 1
                        if item.dim() == 4 and item.size(1) in [1, 3]:
                            sample_images = item
                        # Labels are typically 1D or 2D with smaller values
                        elif item.dim() <= 2 or (item.dim() == 1):
                            continue  # Skip labels for now
            else:
                sample_images = sample_batch
            
            sample_images = sample_images.to(self.device)
            sample_feat = self.backbone(sample_images)
            sample_feat = self._extract_features(sample_feat)
            feat_dim = sample_feat.size(1)
        
        classifier = torch.nn.Linear(feat_dim, len(self.selected_classes)).to(self.device)
        
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(classifier.parameters()), 
            lr=self.lr
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                # Extract images and labels from batch
                if isinstance(batch, (list, tuple)):
                    images = None
                    labels = None
                    for item in batch:
                        if isinstance(item, torch.Tensor):
                            # Images: 4D tensor [B, C, H, W]
                            if item.dim() == 4 and item.size(1) in [1, 3]:
                                images = item
                            # Labels: 1D or squeezable tensor
                            elif item.dim() <= 2:
                                labels = item
                    
                    if images is None or labels is None:
                        raise ValueError(f"Could not identify images and labels in batch. Batch items shapes: {[item.shape if isinstance(item, torch.Tensor) else type(item) for item in batch]}")
                else:
                    raise ValueError("DataLoader must yield (images, labels) pairs")
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if labels.dim() > 1:
                    labels = labels.squeeze()
                
                emb = self.backbone(images)
                emb = self._extract_features(emb)
                
                logits = classifier(emb)
                loss = self.loss_fn(logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self.backbone, classifier
    
    @torch.no_grad()
    def evaluate_model(
        self, backbone: torch.nn.Module, classifier: torch.nn.Module, 
        test_dataloader: DataLoader
    ) -> float:
        """
        Evaluate the finetuned model on the test set.
        """
        backbone.eval()
        classifier.eval()
        
        correct = 0
        total = 0
        
        for batch in test_dataloader:
            # Extract images and labels from batch
            if isinstance(batch, (list, tuple)):
                images = None
                labels = None
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        # Images: 4D tensor [B, C, H, W]
                        if item.dim() == 4 and item.size(1) in [1, 3]:
                            images = item
                        # Labels: 1D or squeezable tensor
                        elif item.dim() <= 2:
                            labels = item
                
                if images is None or labels is None:
                    raise ValueError(f"Could not identify images and labels in batch")
            else:
                raise ValueError("DataLoader must yield (images, labels) pairs")
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Ensure labels are 1D (squeeze if needed)
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            # Forward pass
            emb = backbone(images)
            emb = self._extract_features(emb)
            
            outputs = classifier(emb)
            _, predicted = torch.max(outputs, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def evaluate(
        self,
        n_samples: Optional[int] = None,
        repeat: int = 5,
    ) -> Tuple[float, float]:
        """
        Train and evaluate a finetuned model with end-to-end backbone training.

        Args:
            n_samples (Optional[int]): Number of samples per class for few-shot learning. If None, use all data.
            repeat (int): Number of times to repeat the evaluation for averaging.

        Returns:
            Tuple[float, float]: Average train and test accuracy.
        """
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("train_loader and test_loader must be provided for end-to-end finetuning")

        # Map and filter labels
        train_feats, train_labels = self._map_labels_and_filter(
            self.train_features, self.train_labels
        )
        test_feats, test_labels = self._map_labels_and_filter(
            self.test_features, self.test_labels
        )

        if n_samples is not None:
            _, _, train_indices = self._sample_fewshot(train_feats, train_labels, n_samples)
            train_loader_to_use = self._create_fewshot_loader(self.train_loader, train_indices)
        else:
            train_loader_to_use = self.train_loader
        
        test_loader_to_use = self.test_loader

        train_accs, test_accs = [], []
        # Saving initial backbone state for independent repeats
        initial_state = copy.deepcopy(self.backbone.state_dict())

        for run in range(repeat):
            print(f"Run {run + 1}/{repeat}")
            
            # Reset backbone and classifier for each run
            # Note: This creates fresh copies to avoid training on already-trained model
            self.backbone.load_state_dict(initial_state)
            finetuned_backbone, classifier = self.finetune_model(train_loader_to_use)

            train_acc = self.evaluate_model(finetuned_backbone, classifier, train_loader_to_use)
            test_acc = self.evaluate_model(finetuned_backbone, classifier, test_loader_to_use)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

        avg_train_acc = np.mean(train_accs)
        avg_test_acc = np.mean(test_accs)

        return float(avg_train_acc), float(avg_test_acc)