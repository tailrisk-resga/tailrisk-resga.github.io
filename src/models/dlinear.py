"""DLinear model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)



class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean



class DLinear(nn.Module):
    def __init__(self, args):
        super(DLinear, self).__init__()
        self.seq_len = args.sequence_length
        self.pred_len = 1  
        self.enc_in = args.feature_dim
        self.hidden_dim = args.hidden_dim

        self.decomposition = SeriesDecomp(kernel_size=25)

        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.pred_len * self.enc_in, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 2)  
        )

        # optional weight init
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
        nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)

    def forward(self, x):
        # x: [B, T, D]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output  # shape: [B, D, 1]
        x = x.permute(0, 2, 1)  # [B, 1, D]
        x = self.fc(x)  # [B, 2]

        v = -F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["DLinear"]
