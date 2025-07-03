import torch
from einops import einsum, rearrange

class LkyEmbedding(torch.nn.Module):
    torch.nn.Embedding
    def __init__(self, num_embeddings, embedding_num, device=None, dtype=None, weights=None): # the weights must be row-major
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_num = embedding_num
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.nn.Parameter(torch.empty(size=(num_embeddings, embedding_num), device=device, dtype=dtype))
            ceta = 1
            torch.nn.init.trunc_normal_(self.weights, mean=0, std=ceta, a=-3 * ceta, b=3 * ceta)
    
    def forward(self, token_ids):
        return self.weights[token_ids]


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

class LkyRMSnorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None, weights=None): # the weights must be row-major
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if weights is not None:
            self.weights = weights
        else:
            print(d_model, device, dtype)
            self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        y = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + torch.tensor([self.eps]))

        assert y.shape[-1] == 1

        y = einsum(1 / y, self.weights,
                   "... l, d_model -> ... d_model"
                    ) * x
        x = x.to(in_dtype)
        return y
