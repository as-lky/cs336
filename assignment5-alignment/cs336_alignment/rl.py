import torch
from typing import Literal
from einops import einsum, rearrange


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
):
    rollout_batch_size = len(rollout_responses)
    raw_rewards = torch.tensor([reward_fn(rollout_responses[i], repeated_ground_truths[i])['reward'] for i in range(rollout_batch_size)], dtype=torch.float32)
    rewards = rearrange(raw_rewards, '(group_num group_size) -> group_num group_size', group_size=group_size)
    if normalize_by_std:
        advantages = (rewards - torch.mean(rewards, dim=-1, keepdim=True)) / (torch.std(rewards, dim=-1, keepdim=True) + advantage_eps)
    else:
        advantages = rewards - torch.mean(rewards, dim=-1, keepdim=True)
    advantages = rearrange(advantages, 'group_num group_size -> (group_num group_size)')
    return advantages, raw_rewards, None

def compute_naive_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs):
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange):
    tmp = torch.exp(policy_log_probs - old_log_probs)
    L = tmp * advantages
    R = torch.clip(tmp, 1 - cliprange, 1 + cliprange) * advantages
    return -torch.minimum(L, R), None

def masked_mean(tensor, mask, dim):
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    if loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), None
    return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), None

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
):
    loss = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)[0]
    loss = torch.mean(masked_mean(tensor=loss, mask=response_mask.int(), dim=-1)) / gradient_accumulation_steps
    loss.backward()
    return loss, None