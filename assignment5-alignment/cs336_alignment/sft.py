import torch
from einops import einsum

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # no tokenizer
    ...

def get_response_log_probs(model, inputs_ids, labels, return_token_entropy=False):
    # no model
    ...
    
def compute_entropy(logits):
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    probabilities = torch.softmax(logits, dim=-1)
    A = torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(probabilities * (logits - A), dim=-1)

def masked_normalize(x, mask, norm_c, dim):
    return torch.sum(x * mask, dim=dim) / norm_c
    