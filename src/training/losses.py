import torch

def L_FZ0(alpha: float,
          y: torch.Tensor, 
          v: torch.Tensor, 
          e: torch.Tensor):
    """FZ Loss Function"""
    indicator = (y <= v).float()
    term1 = - (1 / (alpha * e)) * indicator * (v - y)
    term2 = (v / e) + torch.log(-100 * e) - 1
    loss = term1 + term2
    return torch.mean(loss)

def L_FZ0_smooth(alpha,
                 y: torch.Tensor,
                 v: torch.Tensor,
                 e: torch.Tensor,
                 smoothing: float):
    """Smoothed FZ Loss Function"""
    indicator_smooth = 1 / (1 + torch.exp(torch.clamp(smoothing * 100 * (y - v), max=80.0)))
    term1 = - (1 / (alpha * e)) * indicator_smooth * (v - y)
    term2 = (v / e) + torch.log(-100 * e) - 1
    loss = term1 + term2
    return torch.mean(loss)