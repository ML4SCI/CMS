# losses.py
import torch

def pTLossTorch(y_pred, y_true):
    y_t = (y_true < 80).float() * y_true + (y_true >= 80).float() * (y_true < 250).float() * y_true**2.4 + (y_true >= 160).float() * 10
    return torch.mean(y_t * ((y_pred - y_true) / y_true)**2) / 250

def CustompTLoss(output, target, lower_pt_limit):
    if not isinstance(lower_pt_limit, torch.Tensor):
        lower_pt_limit = torch.tensor(lower_pt_limit)
    
    lower_pt_limit = lower_pt_limit.to(output.dtype)
    output = torch.clip(output, min=lower_pt_limit.to(output.device))
    loss = torch.mean((target - output)**2 + torch.gt(output, lower_pt_limit.long() * 
        (1 / (1 + torch.exp(-(output - lower_pt_limit) * 3)) - 1) + 
        torch.le(output, lower_pt_limit).long() * (-1/2)))
    return loss
