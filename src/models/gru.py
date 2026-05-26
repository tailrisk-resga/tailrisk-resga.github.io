"""GRU model."""

import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
                    input_size = args.feature_dim, 
                    hidden_size = args.hidden_dim, 
                    num_layers = args.num_layers, 
                    batch_first=True, 
                    dropout=args.dropout
                    )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(args.hidden_dim, 2)
        )
        
    
    def forward(self, x):
        gru_out, _ = self.gru(x)  # gru_out shape: (batch, seq_len, hidden_size)
        gru_last_step = gru_out[:, -1, :]  
        x = self.fc_layers(gru_last_step)
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["GRU"]
