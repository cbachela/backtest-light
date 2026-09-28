"""Shared asset-level point predictor for the Wang reproduction."""

from torch import nn


class ReturnPredictor(nn.Module):
    """Architecture and normalization order follow Wang Appendix A.2.1."""

    def __init__(self, features: int, depth: int = 3):
        super().__init__()
        if features < 1 or depth not in range(6):
            raise ValueError("features must be positive; depth must be 0 (linear) through 5")
        layers = []
        if depth:
            layers.append(nn.LayerNorm(features))
            for width in (32, 16, 8, 4, 2)[:depth]:
                layers.extend(
                    [nn.Linear(features, width), nn.ReLU(), nn.BatchNorm1d(width), nn.Dropout(0.05)]
                )
                features = width
        layers.append(nn.Linear(features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        # One complete date cross-section: [assets, features].
        return self.network(features).squeeze(-1)
