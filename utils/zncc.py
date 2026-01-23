import torch

def zncc(pred: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8):
    assert pred.shape == truth.shape
    
    assert pred.dim() == 4
    
    b,t,h,w = pred.shape
    
    x = pred.reshape(b*t, -1).float()
    y = truth.reshape(b*t, -1).float()
    
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    
    num = (x * y).sum(dim=1)
    denorm = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(eps)
    
    rho = (num / denorm).clamp(-1.0, 1.0)
    
    loss = 0.5 * (1.0 - rho)
    
    return loss.mean()