# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Core types and shared advantage computation for AlphaGRPO training.

Reward postprocess is split into two layers:
- Shared layer: `compute_advantages()` here — handles gather → stat_tracker → log → reshape → masking
- Task layer: each task implements its own `reward_postprocess()` that handles task-specific
  reward modifications, then calls `compute_advantages()`.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate.utils import gather, gather_object


# ====== Core Types ======

@dataclass
class GenerationStep:
    """One generation step. Training loop iterates over a list of steps.

    Step types (generation — has loss):
      - "think_text"  : think text (internal reasoning). Only advances gen_context.
      - "text"        : output text (answer). Advances gen; between-step advances cfg_text + cfg_img.
      - "image"       : generated image (SDE sampling). Between-step advances gen + cfg_text.

    Step types (input conditioning — no loss):
      - "input_text"  : user text prompt. cfg_text = deepcopy(gen), gen += text, cfg_img += text.
      - "input_image" : user image input. gen += image, cfg_text = deepcopy(gen).
    """
    step_type: str  # "think_text" | "text" | "image" | "input_text" | "input_image"

    # ---- text fields ----
    texts: Optional[List[str]] = None       # generated text (think text, answer, etc.)
    text_per_token_log_probs: Optional[Any] = None

    # ---- image fields ----
    images: Optional[list] = None
    latents: Optional[Any] = None           # (B, T, ...)
    next_latents: Optional[Any] = None      # (B, T, ...)
    image_log_probs: Optional[Any] = None   # (B, T)
    sde_window: Optional[tuple] = None      # (start, end)
    task_type: Optional[str] = None         # 't2i' | 'edit'
    image_shape: Optional[tuple] = None     # actual rollout shape as (height, width)

    # ---- ref model data (set by compute_kl) ----
    ref_text_per_token_log_probs: Optional[Any] = None
    ref_prev_latents_means: Optional[Any] = None
    ref_model_output: Optional[Any] = None
    ema_model_output: Optional[Any] = None

    # ---- flow-matching cache (lazy-filled by compute_image_logprob) ----
    fm_noise: Optional[List[Any]] = None
    fm_sigma: Optional[List[Any]] = None

    # ---- recompute data (set by recompute_log_prob) ----
    rollout_text_per_token_log_probs: Optional[Any] = None
    rollout_image_log_probs: Optional[Any] = None
    image_outputs: Optional[Any] = None


@dataclass
class AdvantageContext:
    """Context needed by compute_advantages (replaces closure-captured variables)."""
    accelerator: Any
    tokenizer: Any
    global_step: int
    prompt_ids: Any


# ====== Shared Utilities ======

def compute_advantages(config, sample, stat_tracker, ctx: AdvantageContext,
                       prompts_mean=None) -> Tuple[dict, bool]:
    """Shared advantage computation. All tasks call this after task-specific reward modifications.

    Precondition: sample["rewards"]["avg"] has been set/modified by the task's reward_postprocess.

    Steps:
    1. Gather rewards across GPUs
    2. Per-prompt stat tracking → advantages (or simple normalization)
    3. Log metrics (group_size, zero_std_ratio, think_text_length)
    4. Skip check (all advantages == 0)
    5. Reshape advantages for timesteps
    6. text_advantages masking (if think + mask_image_adv_w_negative_think_text)
    """
    from utils import calculate_zero_std_ratio

    should_skip = False
    accelerator = ctx.accelerator

    # 1. Gather rewards
    gathered_rewards_avg = accelerator.gather(sample["rewards"]['avg'].detach().clone()).cpu().numpy()

    # 2. Advantage computation
    if config.per_prompt_stat_tracking:
        gathered_prompts = gather_object(sample['prompts'])

        if prompts_mean is not None and config.train.stage1_reward_as_group_mean:
            advantages = stat_tracker.update(gathered_prompts, gathered_rewards_avg,
                                             group_mean=prompts_mean, type=config.train.algorithm)
        else:
            advantages = stat_tracker.update(gathered_prompts, gathered_rewards_avg)

        group_size, trained_prompt_num = stat_tracker.get_stats()
        zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(gathered_prompts, {'ori_avg': gathered_rewards_avg})

        if config.sample.think:
            think_text_length = np.mean(gather_object([len(t.split(' ')) for t in sample['think_texts']]))
        else:
            think_text_length = 0

        # 3. Log metrics
        if accelerator.is_main_process:
            metrics = {
                "group_size": group_size,
                "trained_prompt_num": trained_prompt_num,
                "zero_std_ratio": zero_std_ratio,
                "reward_std_mean": reward_std_mean,
                "think_text_length": think_text_length,
            }
            if config.logger_type == 'wandb':
                wandb.log(metrics, step=ctx.global_step)
            else:
                accelerator.log(metrics, step=ctx.global_step)

        stat_tracker.clear()

        # Isolated text reward advantages (used by reflection)
        if config.train.isolate_image_text_reward and 'text_avg' in sample['rewards']:
            text_rewards_gathered = gather(sample['rewards']['text_avg']).cpu()
            text_advantages = stat_tracker.update(gathered_prompts, text_rewards_gathered,
                                                  type=config.train.algorithm)
            stat_tracker.clear()
            text_advantages = torch.as_tensor(text_advantages)
            sample["text_advantages"] = (
                text_advantages.reshape(accelerator.num_processes, -1, 1)[accelerator.process_index]
                .to(accelerator.device)
            )

        advantages = torch.as_tensor(advantages)
    else:
        advantages = (gathered_rewards_avg - gathered_rewards_avg.mean()) / (
            gathered_rewards_avg.std() + 1e-4)
        advantages = torch.as_tensor(advantages)

    # 4. Skip check
    if advantages.abs().sum() == 0:
        if accelerator.is_local_main_process:
            print(f"Skipping rollout for step {ctx.global_step} - all advantages are zero")
        sample["is_postprocessed"] = True
        return sample, True

    # 5. Reshape advantages for timesteps
    sample["advantages"] = (
        advantages.reshape(accelerator.num_processes, -1, 1)[accelerator.process_index]
        .to(accelerator.device)
    )
    num_train_timesteps = max(int(config.sample.num_steps * config.train.timestep_fraction), 1)
    sample["advantages"] = sample["advantages"].repeat(1, num_train_timesteps)

    # Debug logging
    if accelerator.is_main_process:
        pad_token_id = tokenizer_pad_id(ctx.tokenizer)
        seq_lengths = (ctx.prompt_ids != pad_token_id).sum(dim=1)
        batch_sum_seq_length = seq_lengths.sum().item()

        print('rewards', gathered_rewards_avg)
        print('advantages', advantages)
        if config.sample.think:
            print('think_text', sample['think_texts'])

    # 6. text_advantages masking
    # Preserve isolated text advantages if already computed by isolate_image_text_reward;
    # otherwise default to the shared advantages (when think is enabled) or None.
    if "text_advantages" not in sample or sample["text_advantages"] is None:
        if config.sample.think:
            sample["text_advantages"] = sample["advantages"]
        else:
            sample["text_advantages"] = None

    if config.sample.think and config.train.mask_image_adv_w_negative_think_text:
        negative_think_mask = sample['rewards']['think_text_reward'] < 0

        if config.train.reflect_think_text_format:
            negative_think_mask = negative_think_mask | (sample['rewards']['reflective_think_text_reward'] < 0)

        sample["advantages"] = sample["advantages"] * (~negative_think_mask).unsqueeze(-1).float()

    sample["is_postprocessed"] = True
    return sample, should_skip


def tokenizer_pad_id(tokenizer):
    """Get pad token id from tokenizer, defaulting to 0."""
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return 0