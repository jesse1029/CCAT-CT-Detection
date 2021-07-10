import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(n_features//2, n_features//4),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(n_features//4, n_classes))

    def forward(self, x):
        return self.model(x)
