import torch
import math
from einops import rearrange, einsum

class LkyAdamW(torch.optim.Optimizer):
    def __init__(self, params=None, lr=1e-3, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr, "beta1":betas[0], "beta2":betas[1], "epsilon":eps, "lam":weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]
            lam = group["lam"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data, dtype=torch.float32))
                v = state.get("v", torch.zeros_like(p.data, dtype=torch.float32))
                t = state.get("t", 1)
                state["m"] = beta1 * m + (1 - beta1) * p.grad.data
                state["v"] = beta2 * v + (1 - beta2) * p.grad.data * p.grad.data
                lr_now = lr * math.sqrt(1 - (beta2 ** t)) / (1 - (beta1 ** t))
                p.data -= lr_now * state["m"] / (torch.sqrt(state["v"]) + epsilon)
                p.data -= p.data * lr * lam
                state["t"] = t + 1
        return loss