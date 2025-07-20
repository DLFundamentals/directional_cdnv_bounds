import torch
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, loader):
        """
        Extract (h, g(h)) features and labels from a dataloader.
        Returns:
            [features_h, features_gh], labels
        """
        feats_h, feats_gh, labels = [], [], []

        for batch in tqdm(loader, desc="Extracting Features"):
            _, x, y = batch
            x = x.to(self.device)
            h, g_h = self.model(x)
            feats_h.append(h.view(h.size(0), -1).cpu())
            feats_gh.append(g_h.view(g_h.size(0), -1).cpu())
            labels.append(y.cpu())

        features_h = torch.cat(feats_h, dim=0)
        features_gh = torch.cat(feats_gh, dim=0)
        labels = torch.cat(labels, dim=0)

        return [features_h, features_gh], labels
