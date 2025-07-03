import torch
from einops import einsum

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