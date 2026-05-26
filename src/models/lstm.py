"""LSTM model."""

import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
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
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, hidden_size)
        lstm_last_step = lstm_out[:, -1, :]  
        x = self.fc_layers(lstm_last_step)
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["LSTM"]
