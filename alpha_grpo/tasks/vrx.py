# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""BAGEL image-only visual chain-of-thought task for VR-X."""

from __future__ import annotations

import contextlib
import re
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import torch
from PIL import Image

from common import AdvantageContext, GenerationStep, compute_advantages
from tasks import BagelBaseTask, register
from utils import create_generator, log_images_to_tracker, sample_sde_window


def resolve_num_reasoning_steps(config, prompt, gt_trajectory):
    vrx = config.vrx
    if getattr(vrx, "use_gt_step_count", False) and gt_trajectory:
        count = len(gt_trajectory)
    elif getattr(vrx, "use_instruction_step_count", False):
        match = re.search(r"\b(\d+)\s+(?:steps?|frames?|images?)\b", prompt, re.I)
        count = int(match.group(1)) if match else int(vrx.num_reasoning_steps)
    else:
        count = int(vrx.num_reasoning_steps)
    maximum = int(getattr(vrx, "max_reasoning_steps", count))
    return max(1, min(count, maximum))


def build_image_generation_step(images, latents, image_log_probs, sde_window, task_type="edit"):
    image_shapes = {image.size[::-1] for image in images}
    if len(image_shapes) != 1:
        raise ValueError(f"All generated images in a BAGEL batch need one shape: {image_shapes}")
    latents = torch.stack(latents, dim=1)
    if len(image_log_probs):
        image_log_probs = torch.stack(image_log_probs, dim=1)
    return GenerationStep(
        step_type="image",
        images=images,
        latents=latents[:, :-1],
        next_latents=latents[:, 1:],
        image_log_probs=image_log_probs,
        sde_window=sde_window,
        task_type=task_type,
        image_shape=next(iter(image_shapes)),
    )


def _trajectory_grid(input_images, gt_images, generated_images, height=192):
    frames = list(input_images) + list(gt_images) + list(generated_images)
    if not frames:
        return Image.new("RGB", (height, height), "white")
    resized = []
    for image in frames:
        image = image.convert("RGB")
        width = max(1, round(image.width * height / image.height))
        resized.append(image.resize((width, height)))
    canvas = Image.new("RGB", (sum(image.width for image in resized), height), "white")
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


@register("vrx")
class VRXTask(BagelBaseTask):

    def rollout(self, config, tokenizer, batch_data, global_step, stat_tracker, **kwargs):
        if len(batch_data) != 4:
            raise RuntimeError("VRXTask expects prompts, metadata, input images, and GT trajectories")
        instructions, metadata, input_images, gt_trajectories = batch_data
        step_counts = [
            resolve_num_reasoning_steps(config, prompt, gt)
            for prompt, gt in zip(instructions, gt_trajectories)
        ]
        if len(set(step_counts)) != 1:
            raise ValueError(
                "A BAGEL rollout batch must use one visual-CoT length; use local grouped sampling "
                f"or fixed steps. Got {step_counts}."
            )
        num_steps = step_counts[0]
        if self.accelerator.is_local_main_process:
            if getattr(config.vrx, "use_gt_step_count", False):
                step_strategy = "gt_trajectory_length"
            elif getattr(config.vrx, "use_instruction_step_count", False):
                step_strategy = "instruction_or_fixed_fallback"
            else:
                step_strategy = "fixed"
            print(f"VR-X rollout: visual_steps={num_steps}, strategy={step_strategy}")
        input_counts = {len(images) for images in input_images}
        if len(input_counts) != 1:
            raise ValueError(f"All VR-X samples in a BAGEL batch need equal input-frame counts: {input_counts}")

        autocast = contextlib.nullcontext if config.use_lora else self.accelerator.autocast
        sde_window = sample_sde_window(config)
        input_lists = [[prompt, *images] for prompt, images in zip(instructions, input_images)]
        generation_steps = []
        trajectory_steps = []
        initial_contexts = None

        with autocast():
            with torch.no_grad():
                for visual_step in range(num_steps):
                    generators = None
                    if config.train.init_same_noise:
                        group_uids = [item["uid"] for item in metadata]
                        generators = create_generator(
                            [f"{uid}:visual_step_{visual_step}" for uid in group_uids], global_step
                        )
                    outputs = self.generate_image(
                        input_lists,
                        config,
                        think=False,
                        sde_window=sde_window,
                        generators=generators,
                        deterministic=False,
                        use_flowcps=config.sample.use_flowcps,
                        task="edit" if input_counts != {0} or visual_step > 0 else "t2i",
                        return_log_probs=True,
                    )
                    images, latents, image_log_probs, think_texts, text_log_probs, contexts = outputs
                    if think_texts is not None or text_log_probs is not None:
                        raise RuntimeError("VR-X visual reasoning must not generate textual CoT")
                    if initial_contexts is None:
                        initial_contexts = contexts
                    generation_steps.append(
                        build_image_generation_step(
                            images, latents, image_log_probs, sde_window,
                            task_type="edit" if input_counts != {0} or visual_step > 0 else "t2i",
                        )
                    )
                    trajectory_steps.append(images)
                    for sequence, image in zip(input_lists, images):
                        sequence.append(image)

        trajectories = [
            [trajectory_steps[step][index] for step in range(num_steps)]
            for index in range(len(instructions))
        ]
        group_uids = [item["uid"] for item in metadata]
        prompt_ids = tokenizer(
            instructions,
            padding="max_length",
            max_length=config.model.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device, dtype=torch.long)

        reward_input = {
            "trajectories": trajectories,
            "prompts": instructions,
            "metadata": metadata,
            "input_images": input_images,
            "gt_trajectories": gt_trajectories,
            "group_uids": group_uids,
            "expected_num_frames": step_counts,
        }
        rewards_future = self.compute_rewards(reward_input, return_future=True)
        sample = {
            "prompt_ids": prompt_ids,
            # Advantage grouping must use stable sample identity, not possibly duplicated text.
            "prompts": group_uids,
            "instructions": instructions,
            "prompt_metadata": metadata,
            "contexts": initial_contexts,
            "is_postprocessed": False,
            "input_images": input_images,
            "gt_trajectories": gt_trajectories,
            "trajectories": trajectories,
            "images": trajectory_steps[-1],
            "generation_steps": generation_steps,
            "group_uids": group_uids,
            "num_visual_steps": num_steps,
            "task": "vrx",
            "rewards_future": rewards_future,
        }
        ctx = AdvantageContext(
            accelerator=self.accelerator,
            tokenizer=tokenizer,
            global_step=global_step,
            prompt_ids=prompt_ids,
        )
        sample["reward_postprocess_fn"] = partial(self.reward_postprocess, ctx=ctx)
        if config.train.async_reward_fn:
            return sample, False
        return sample["reward_postprocess_fn"](config, sample, stat_tracker=stat_tracker)

    @classmethod
    def reward_postprocess(cls, config, sample, stat_tracker, ctx=None):
        if sample["is_postprocessed"]:
            return sample, False
        rewards, reward_metadata = sample.pop("rewards_future").result()
        sample["rewards"] = {
            key: torch.as_tensor(value, device=ctx.accelerator.device).float()
            for key, value in rewards.items()
        }
        sample["reward_metadata"] = reward_metadata
        sample["rewards"]["ori_avg"] = sample["rewards"]["avg"].detach().clone()
        return compute_advantages(config, sample, stat_tracker, ctx)

    def collect_metrics(self, sample, info):
        for key, value in sample["rewards"].items():
            info[f"reward/{key}"].append(torch.mean(value))
        rewards = sample["rewards"]["avg"]
        groups = defaultdict(list)
        for uid, reward in zip(sample["group_uids"], rewards):
            groups[uid].append(float(reward.detach().cpu()))
        group_stds = [np.std(values) for values in groups.values()]
        info["group/reward_mean"].append(rewards.mean())
        info["group/reward_std"].append(torch.tensor(np.mean(group_stds), device=rewards.device))
        info["group/zero_std_ratio"].append(torch.tensor(
            np.mean([std < 1e-8 for std in group_stds]), device=rewards.device
        ))
        info["rollout/num_visual_steps"].append(torch.tensor(
            sample["num_visual_steps"], device=rewards.device, dtype=torch.float32
        ))
        info["rollout/group_size"].append(torch.tensor(
            np.mean(list(Counter(sample["group_uids"]).values())),
            device=rewards.device,
            dtype=torch.float32,
        ))

    def log_training_step(self, train_samples, accelerator, global_step, config, prefix=""):
        sample = train_samples[0]
        grids = [
            _trajectory_grid(inputs, gt, generated)
            for inputs, gt, generated in zip(
                sample["input_images"], sample["gt_trajectories"], sample["trajectories"]
            )
        ]
        log_images_to_tracker(
            images=grids,
            prompts=sample["instructions"],
            rewards=sample["rewards"],
            accelerator=accelerator,
            global_step=global_step,
            config=config,
            max_images=config.max_images_per_log,
            prefix="train_vrx",
        )

    @staticmethod
    def get_log_images(sample):
        return [
            _trajectory_grid(inputs, gt, generated)
            for inputs, gt, generated in zip(
                sample["input_images"], sample["gt_trajectories"], sample["trajectories"]
            )
        ]

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        """UniVR released validation: format gate plus absolute trajectory VLM."""
        del kwargs
        accelerator = self.accelerator
        if self.val_reward_fn is None:
            if accelerator.is_local_main_process:
                print("VR-X validation reward is not configured; skipping evaluation.")
            return {}

        if accelerator.is_local_main_process:
            print(
                "Start VR-X validation with single-sample absolute VLM reward "
                f"and {config.vrx.num_reasoning_steps} visual steps."
            )

        model = self.inferencer.model
        model.eval()
        pending = []
        with torch.no_grad():
            for batch_data in test_dataloader:
                instructions, metadata, input_images, gt_trajectories = batch_data
                step_counts = [
                    resolve_num_reasoning_steps(config, prompt, gt)
                    for prompt, gt in zip(instructions, gt_trajectories)
                ]
                if len(set(step_counts)) != 1:
                    raise ValueError(
                        "A VR-X validation batch must use one visual trajectory length: "
                        f"{step_counts}"
                    )

                num_steps = step_counts[0]
                input_lists = [
                    [prompt, *images]
                    for prompt, images in zip(instructions, input_images)
                ]
                trajectory_steps = []
                with autocast():
                    for visual_step in range(num_steps):
                        images = self.generate_image(
                            input_lists,
                            config,
                            think=False,
                            num_timesteps=config.sample.eval_num_steps,
                            deterministic=True,
                            task="edit",
                            return_log_probs=False,
                        )
                        trajectory_steps.append(images)
                        for sequence, image in zip(input_lists, images):
                            sequence.append(image)

                trajectories = [
                    [trajectory_steps[step][index] for step in range(num_steps)]
                    for index in range(len(instructions))
                ]
                group_uids = [item["uid"] for item in metadata]
                reward_input = {
                    "trajectories": trajectories,
                    "prompts": instructions,
                    "metadata": metadata,
                    "input_images": input_images,
                    "gt_trajectories": gt_trajectories,
                    "group_uids": group_uids,
                    "expected_num_frames": step_counts,
                }
                future = self.executor.submit(self.val_reward_fn, reward_input)
                grids = [
                    _trajectory_grid(inputs, gt, generated)
                    for inputs, gt, generated in zip(
                        input_images, gt_trajectories, trajectories
                    )
                ]
                pending.append((future, instructions, grids))

        all_rewards = []
        last_prompts = []
        last_grids = []
        for future, instructions, grids in pending:
            rewards, _ = future.result()
            tensor_rewards = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            all_rewards.append(tensor_rewards)
            last_prompts = instructions
            last_grids = grids

        if not all_rewards:
            model.train()
            return {}

        aggregated = {}
        for key in all_rewards[0]:
            local_values = torch.cat([reward[key] for reward in all_rewards])
            if hasattr(accelerator, "gather_for_metrics"):
                values = accelerator.gather_for_metrics(local_values)
            else:
                values = accelerator.gather(local_values)
            aggregated[key] = values.detach().cpu().numpy()

        if accelerator.is_main_process:
            accelerator.log(
                {f"eval/{key}": float(value.mean()) for key, value in aggregated.items()},
                step=global_step,
            )

        if last_grids:
            log_images_to_tracker(
                images=last_grids,
                prompts=last_prompts,
                rewards=all_rewards[-1],
                accelerator=accelerator,
                global_step=global_step,
                config=config,
                max_images=config.max_images_per_log,
                prefix="eval_vrx",
            )

        model.train()
        if accelerator.is_local_main_process:
            print(
                "VR-X validation results:",
                {key: float(value.mean()) for key, value in aggregated.items()},
            )
        return aggregated

