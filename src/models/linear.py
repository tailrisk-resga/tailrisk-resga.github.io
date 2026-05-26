"""Linear model."""

import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, args):
        super(Linear, self).__init__()
        self.linear = nn.Linear(args.feature_dim, 2)
        self.relu = nn.ReLU()
        

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        if self.linear.bias is not None:
            self.linear.bias.data.normal_(0, 0.01)
            
    def forward(self, x):
        x = self.linear(x)
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["Linear"]
