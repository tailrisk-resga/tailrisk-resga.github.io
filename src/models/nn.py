"""Feed-forward neural network model."""

import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, args):
        super(NN, self).__init__()
        
        # r = (2 / args.hidden_dim) ** (1 / 3)
 
        r = 0.5
        self.fc_layers = nn.Sequential(
            nn.Linear(args.feature_dim, args.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Dropout(args.dropout),

            nn.Linear(args.hidden_dim, int(args.hidden_dim * r)),
            nn.ReLU(),
            nn.BatchNorm1d(int(args.hidden_dim * r)),
            nn.Dropout(args.dropout),

            nn.Linear(int(args.hidden_dim * r), int(args.hidden_dim * r * r)),
            nn.ReLU(),
            nn.BatchNorm1d(int(args.hidden_dim * r * r)),
            nn.Dropout(args.dropout),

            nn.Linear(int(args.hidden_dim * r * r), 2)
        )
        
    def forward(self, x):
        x = self.fc_layers(x)
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["NN"]
