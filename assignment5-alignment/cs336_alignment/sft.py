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
    
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=None) / gradient_accumulation_steps / policy_log_probs.shape[0] # / batch_size   # WRONG! can use torch.mean()!!!
    loss.backward()
    return loss, None