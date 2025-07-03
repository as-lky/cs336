import torch
from einops import einsum

class LkyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, weights=None): # the weights must be row-major
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.nn.Parameter(torch.empty(size=(in_features, out_features), device=device, dtype=dtype))
            ceta = (2.0 / (in_features + out_features)) ** 0.5
            torch.nn.init.trunc_normal_(self.weights, mean=0, std=ceta, a=-3 * ceta, b=3 * ceta)
    
    def forward(self, x):
        return einsum(x, self.weights, "... d_in, d_in d_out -> ... d_out")
