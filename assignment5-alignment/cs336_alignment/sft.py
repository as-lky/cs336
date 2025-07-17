import torch
from einops import einsum

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    ...
    
def compute_entropy(logits):
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    probabilities = torch.softmax(logits, dim=-1)
    A = torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(probabilities * (logits - A), dim=-1)