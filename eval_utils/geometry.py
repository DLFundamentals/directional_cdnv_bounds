import torch

class GeometricEvaluator:
    def __init__(self, num_classes=10, device='cuda'):
        self.num_classes = num_classes
        self.device = device

    def compute_class_means(self, features, labels):
        means = []
        for c in range(self.num_classes):
            idxs = (labels == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                means.append(None)
                continue
            class_feats = features[idxs]
            means.append(class_feats.mean(dim=0))
        return means

    def compute_class_second_moments(self, features, labels):
        moments = []
        for c in range(self.num_classes):
            idxs = (labels == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                moments.append(None)
                continue
            class_feats = features[idxs]
            moments.append((class_feats ** 2).mean(dim=0))
        return moments

    def compute_cdnv(self, features, labels):
        features = features.to(self.device)
        labels = labels.to(self.device)

        means = self.compute_class_means(features, labels)
        moments = self.compute_class_second_moments(features, labels)

        cdnv_total = 0.0
        num_pairs = 0

        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if means[i] is None or means[j] is None:
                    continue

                dist_sq = torch.norm(means[i] - means[j])**2
                # variance = E[x^2] - (E[x])^2, computed as mean_s - mean**2
                var_i = (moments[i] - means[i]**2).sum()
                var_j = (moments[j] - means[j]**2).sum()
                avg_var = 0.5 * (var_i + var_j)

                cdnv_total += (avg_var / dist_sq)
                num_pairs += 1

        return (cdnv_total / num_pairs).item() if num_pairs > 0 else None

    def compute_directional_cdnv(self, features, labels, means=None):
        features = features.to(self.device)
        labels = labels.to(self.device)

        if means is None:
            means = self.compute_class_means(features, labels)

        dir_cdnv_total = 0.0
        num_pairs = 0

        for i in range(self.num_classes):
            idxs_i = (labels == i).nonzero(as_tuple=True)[0]
            if len(idxs_i) == 0:
                continue
            feats_i = features[idxs_i]

            for j in range(self.num_classes):
                if i == j or means[j] is None or means[i] is None:
                    continue

                v = means[j] - means[i]
                v_norm = torch.norm(v)
                if v_norm == 0:
                    continue

                v_hat = v / v_norm
                projections = (feats_i - means[i]) @ v_hat
                dir_var = torch.mean(projections ** 2)
                dir_cdnv = dir_var / (v_norm ** 2)

                dir_cdnv_total += dir_cdnv
                num_pairs += 1

        return (dir_cdnv_total / num_pairs).item() if num_pairs > 0 else None
