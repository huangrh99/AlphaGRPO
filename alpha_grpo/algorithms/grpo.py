# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""GRPO (Group Relative Policy Optimization) algorithm."""

import torch
from algorithms import register, BaseAlgorithm


@register('grpo')
class GRPOAlgorithm(BaseAlgorithm):

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info,
                          gen_step=None,
                          text_per_token_log_probs=None, text_entropies=None, text_token_lens=None,
                          get_high_entropy_mask=None,
                          **kwargs):
        # Extract rollout/ref data from gen_step
        old_text_lp = gen_step.text_per_token_log_probs

        if config.train.true_ratio:
            text_per_token_log_ratio = text_per_token_log_probs - old_text_lp
        else:
            text_per_token_log_ratio = text_per_token_log_probs - text_per_token_log_probs.detach()

        advantages = sample["text_advantages"] if 'text_advantages' in sample else sample["advantages"]
        advantages = advantages[:, 0]
        if config.train.text_importance_sampling_level == "token":
            text_log_ratio = text_per_token_log_ratio
            text_advantages = torch.repeat_interleave(advantages.to(accelerator.device), text_token_lens[:advantages.shape[0]].to(accelerator.device))
        elif config.train.text_importance_sampling_level == "sequence":
            text_advantages = advantages.to(accelerator.device)
            text_log_ratio = torch.stack([item.mean() for item in text_per_token_log_ratio.split(text_token_lens[:advantages.shape[0]].tolist())])
        else:
            raise ValueError(
                f"Unknown importance sampling level: {config.train.text_importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        text_ratio = torch.exp(text_log_ratio)
        coef_1 = text_ratio
        coef_2 = torch.clamp(coef_1, 1 - config.train.text_clip_range, 1 + config.train.text_clip_range_high)

        text_per_token_loss1 = coef_1 * text_advantages.unsqueeze(1)
        text_per_token_loss2 = coef_2 * text_advantages.unsqueeze(1)
        text_policy_loss = -torch.min(text_per_token_loss1, text_per_token_loss2)

        # entropy penalty
        entropy_mask = None
        if config.train.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(text_entropies, 1 - config.train.top_entropy_quantile)
            text_policy_loss = text_policy_loss * entropy_mask

        if config.train.rollout_correction:
            weight = torch.exp(
                text_per_token_log_probs.detach().to(accelerator.device) - old_text_lp).clamp(
                max=config.train.correction_truncated_threshold)
            text_policy_loss = weight * text_policy_loss

        if config.train.text_loss_level == 'token':
            text_policy_loss = torch.mean(text_policy_loss)
        elif config.train.text_loss_level == 'sequence':
            if config.train.text_importance_sampling_level == "token":
                text_policy_loss = torch.stack([item.mean() for item in text_policy_loss.split(text_token_lens[:advantages.shape[0]].tolist())])
            text_policy_loss = torch.mean(text_policy_loss)

        if config.train.text_beta > 0:
            ref_text_lp = gen_step.ref_text_per_token_log_probs
            per_token_k3 = (
                torch.exp(ref_text_lp - text_per_token_log_probs) -
                (ref_text_lp - text_per_token_log_probs) - 1
            )

            importance_ratio = text_ratio.detach()

            if config.train.text_importance_sampling_level == "sequence":
                seq_k3_values = [
                    item.mean()
                    for item in per_token_k3.split(text_token_lens[:advantages.shape[0]].tolist())
                ]
                seq_k3_values = torch.stack(seq_k3_values)
                text_kl_loss = importance_ratio * seq_k3_values
                text_kl_loss = torch.mean(text_kl_loss)
            else:
                text_kl_loss = importance_ratio * per_token_k3
                text_kl_loss = torch.mean(text_kl_loss)

            step_loss = text_policy_loss + config.train.text_beta * text_kl_loss
        else:
            step_loss = text_policy_loss

        # Record metrics
        info["text_policy_loss"].append(text_policy_loss.detach())
        info["text_clipfrac"].append(
            torch.mean(((1.0 - text_ratio) > config.train.text_clip_range).float())
        )
        info["text_clipfrac_high"].append(
            torch.mean(((text_ratio - 1.0) > config.train.text_clip_range_high).float())
        )
        info["text_entropies"].append(text_entropies.mean())
        info["text_ratio"].append(text_ratio.mean().detach())
        print('text_ratio', text_ratio.mean().detach())

        if config.train.text_beta > 0:
            info["text_kl_loss"].append(text_kl_loss.detach())

        return info, step_loss

    @classmethod
    def compute_image_loss(cls, config, sample, accelerator, info,
                           gen_step=None, j=0,
                           image_log_probs=None, prev_latents_mean=None, std_dev_t=None,
                           **kwargs):
        # Extract rollout/ref data from gen_step
        old_image_lp_j = gen_step.image_log_probs[:, j]

        advantages = torch.clamp(
            sample["advantages"][:, j], -config.train.adv_clip_max, config.train.adv_clip_max
        )

        if config.train.true_ratio:
            image_ratio = torch.exp(image_log_probs - old_image_lp_j.detach())
        else:
            image_ratio = torch.exp(image_log_probs - image_log_probs.detach())
        image_ratio = image_ratio.to(accelerator.device)

        image_unclipped_loss = -advantages * image_ratio
        image_clipped_loss = -advantages * torch.clamp(
            image_ratio, 1.0 - config.train.image_clip_range,
            1.0 + getattr(config.train, "image_clip_range_high", config.train.image_clip_range)
        )
        image_policy_loss = torch.maximum(image_unclipped_loss, image_clipped_loss)

        if config.train.rollout_correction:
            weight = torch.exp(image_log_probs.detach() - old_image_lp_j).clamp(
                max=config.train.correction_truncated_threshold)
            weight = weight.to(accelerator.device)
            if accelerator.is_main_process:
                print('correction image weight', torch.exp(image_log_probs.detach() - old_image_lp_j))
            image_policy_loss = weight * image_policy_loss

        image_policy_loss = torch.mean(image_policy_loss)

        if config.train.image_beta > 0:
            ref_prev_mean_j = gen_step.ref_prev_latents_means[:, j]
            image_kl_loss = ((prev_latents_mean - ref_prev_mean_j) ** 2).mean() / (2 * std_dev_t ** 2)
            image_kl_loss = torch.mean(image_kl_loss)
            step_loss = image_policy_loss + config.train.image_beta * image_kl_loss
        else:
            step_loss = image_policy_loss

        info["image_approx_kl"].append(
            0.5 * torch.mean((image_log_probs.detach() - old_image_lp_j) ** 2)
        )
        image_clip_high = getattr(
            config.train, "image_clip_range_high", config.train.image_clip_range
        )
        info["image_clipfrac"].append(torch.mean((
            (image_ratio < 1.0 - config.train.image_clip_range)
            | (image_ratio > 1.0 + image_clip_high)
        ).float()))
        info["image_policy_loss"].append(image_policy_loss.detach())
        info["image_ratio"].append(image_ratio.mean().detach())

        if config.train.image_beta > 0:
            info["image_kl_loss"].append(image_kl_loss.detach())
            ref_kl_metric = image_kl_loss.detach()
        else:
            ref_kl_metric = torch.zeros_like(image_policy_loss.detach())
        info["train/image_ref_kl"].append(ref_kl_metric)

        info["train/image_policy_loss"].append(image_policy_loss.detach())
        info["train/image_ratio"].append(image_ratio.mean().detach())
        info["train/image_clipfrac"].append(info["image_clipfrac"][-1])
        info["train/image_approx_kl"].append(info["image_approx_kl"][-1])
        return info, step_loss
