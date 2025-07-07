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
    def __init__(self, in_features, out_features, device=None, dtype=None, weights=None): # the weights must be col-major (d_out d_in)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weights is not None:
            self.weights = rearrange(weights, "d_out d_in -> d_in d_out")
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

class LkyFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, w1_weights=None, w2_weights=None, w3_weights=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = (8 * d_model // (3 * 64) + 1 ) * 64
        self.w1 = LkyLinear(d_model, d_ff, device, dtype, w1_weights)
        self.w2 = LkyLinear(d_ff, d_model, device, dtype, w2_weights)
        self.w3 = LkyLinear(d_model, d_ff, device, dtype, w3_weights)

    def forward(self, x):
        y = self.w1(x)
        return self.w2(y * torch.sigmoid(y) * self.w3(x))

class LkyRoPE(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert not (d_k & 1)
        self.theta = theta
        self.d_k = d_k,
        self.max_seq_len = max_seq_len
        frac_b = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        frac_t = torch.arange(max_seq_len)
        buffer = einsum(frac_b, frac_t, "d_k_2, seq_len -> seq_len d_k_2")
        buffer.to(device)
        self.register_buffer('weights', buffer, persistent=False)

    def forward(self, x, token_positions):
        weights = self.get_buffer('weights')
        cos = torch.cos(weights)[token_positions]
        sin = torch.sin(weights)[token_positions]
        y = rearrange(x, '... (d_k_2 l) -> ... d_k_2 l', l=2)
        return rearrange(torch.stack(([y[..., 0] * cos - y[..., 1] * sin,
                    y[..., 0] * sin + y[..., 1] * cos]), dim=-1), '... d_k_2 l -> ... (d_k_2 l)', l=2)


class LkySoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, dim):
        y = x - torch.max(x, dim=dim, keepdim=True).values
        return torch.exp(y) / torch.sum(torch.exp(y), dim=dim, keepdim=True)


def lkysoftmax(in_features, dim):
    return LkySoftmax()(in_features, dim)

def lkyattention(Q, K, V, mask=None):
    # no need for 'attention' class?

    assert K.shape[-1] == V.shape[-1]
    d_k = Q.shape[-1]
    tmp = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
       tmp = torch.where(~mask, float('-inf'), tmp)
    return einsum(lkysoftmax(tmp / (d_k ** 0.5), dim=-1) , V, '... queries keys, ... keys d_v -> ... queries d_v')


class LkyMultiheadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_k_v, d_in, seq_len, q_weights=None, k_weights=None, v_weights=None, o_weights=None, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k_v = d_k_v
        self.d_in = d_in
        self.seq_len = seq_len
        self.q = LkyLinear(d_in, d_k_v, device=device, dtype=torch.float32, weights=q_weights)
        self.k = LkyLinear(d_in, d_k_v, device=device, dtype=torch.float32, weights=k_weights)
        self.v = LkyLinear(d_in, d_k_v, device=device, dtype=torch.float32, weights=v_weights)
        self.o = LkyLinear(d_k_v, d_model, device=device, dtype=torch.float32, weights=o_weights)

    def forward(self, x):
        d_k = q_proj_weight.shape[-2]
        d_v = v_proj_weight.shape[-2]
        assert d_k == d_v
        seq_len = in_features.shape[-2]

        big_matrix = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=-2)
        big_matrix = einsum(big_matrix, in_features, '... big_dim d_in, ... seq_len d_in -> ... seq_len big_dim')
        
        q_proj, k_proj, v_proj = big_matrix[..., 0:d_k], big_matrix[..., d_k:(2*d_k)], big_matrix[..., (2*d_k):]
        
        q_proj = rearrange(q_proj, '... seq_len (num_heads else) -> num_heads ... seq_len else', num_heads=num_heads)
        k_proj = rearrange(k_proj, '... seq_len (num_heads else) -> num_heads ... seq_len else', num_heads=num_heads)
        v_proj = rearrange(v_proj, '... seq_len (num_heads else) -> num_heads ... seq_len else', num_heads=num_heads)
        mask_now = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
        now = run_scaled_dot_product_attention(q_proj, k_proj, v_proj, mask_now)
        now = rearrange(now, 'num_heads ... else -> ... (num_heads else)')
        