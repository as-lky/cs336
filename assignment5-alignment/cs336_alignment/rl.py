import torch
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