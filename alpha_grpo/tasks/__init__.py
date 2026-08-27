# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Task hub for AlphaGRPO training.

Usage:
    from tasks import get_task
    TaskClass = get_task(config.train.task)
    task = TaskClass(inferencer=inferencer, accelerator=accelerator, ...)
    output = task.rollout(config, tokenizer, ...)

To add a new task:
    1. Create tasks/my_task.py
    2. Implement @register('my_task') class inheriting BagelBaseTask with rollout, eval
    3. Add `from . import my_task` below
"""

import contextlib

from trl.models import unwrap_model_for_generation


REGISTRY = {}


def register(name):
    """Support registering one task under multiple names: @register(('t2i', 'ti2i'))"""
    def decorator(cls):
        names = name if isinstance(name, (list, tuple)) else [name]
        for n in names:
            REGISTRY[n] = cls
        return cls
    return decorator


def get_task(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]


class BaseTask:
    """Abstract base for all tasks. Pure interface — no state.

    Subclasses that support generation must implement the generation methods
    (generate_image, generate_text, compute_rewards).
    See BagelBaseTask for the Bagel-specific implementation.
    """

    # ---- Core lifecycle (must implement) ----

    def rollout(self, config, tokenizer, batch_data,
                global_step, stat_tracker, **kwargs):
        raise NotImplementedError

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        raise NotImplementedError

    # ---- Generation interface (implement in model-specific base class) ----

    def generate_image(self, input_lists, config, return_log_probs=False, **kwargs):
        """Image generation. return_log_probs=False: ODE → images; True: SDE → 6-tuple."""
        raise NotImplementedError

    def generate_text(self, input_lists, config, return_log_probs=False, **kwargs):
        """Text generation. return_log_probs=False: → (texts, think_texts); True: → 5-tuple."""
        raise NotImplementedError

    def compute_rewards(self, input_data, return_future=False):
        """Reward computation (sync or async)."""
        raise NotImplementedError

    # ---- Logging (override as needed) ----

    def collect_metrics(self, sample, info):
        """Collect per-sample reward metrics into info dict. Called each training sample."""
        for key, value in sample["rewards"].items():
            info[f"reward_{key}"].append(torch.mean(value))

    def log_training_step(self, train_samples, accelerator, global_step, config, prefix=""):
        """Log training images/rewards to wandb. Called every log_image_freq steps.
        prefix: appended to wandb key, e.g. prefix="t2i" → "train_t2i/sample_images".
        Override per task. Default: no-op (tasks without images)."""
        pass

    @staticmethod
    def get_log_sample_index(train_samples):
        """Return the index into train_samples to use for logging. Override if needed."""
        return 0

    @staticmethod
    def get_log_images(sample):
        """Return images list for wandb logging. Override to customize."""
        return sample.get('images', [])


class BagelBaseTask(BaseTask):
    """Bagel-specific base task holding model references and generation methods.

    Swap this base class to support a different model backend.
    """

    def __init__(self, inferencer, accelerator, executor,
                 reward_fn=None, text_reward_fn=None, val_reward_fn=None):
        self.inferencer = inferencer
        self.accelerator = accelerator
        self.executor = executor
        self.reward_fn = reward_fn
        self.text_reward_fn = text_reward_fn
        self.val_reward_fn = val_reward_fn

    def _init_kwargs(self):
        """Return kwargs dict for constructing sibling task instances."""
        return dict(
            inferencer=self.inferencer,
            accelerator=self.accelerator,
            executor=self.executor,
            reward_fn=self.reward_fn,
            text_reward_fn=self.text_reward_fn,
            val_reward_fn=self.val_reward_fn,
        )

    def generate_image(self, input_lists, config, think=False,
                       num_timesteps=None, sde_window=None, generators=None,
                       task='t2i', return_log_probs=False, **kwargs):
        """Image generation via batch_generate_images.

        return_log_probs=False (ODE): returns images.
        return_log_probs=True  (SDE): returns (images, latents, image_log_probs,
                                       think_texts, text_per_token_log_probs, contexts).
        """
        # Branch-specific defaults
        if return_log_probs:
            gen_kwargs = dict(
                return_log_probs=True,
                return_middle_context=True,
                sde_window=sde_window,
                noise_level=config.sample.noise_level,
                generators=generators,
                num_timesteps=config.sample.num_steps if num_timesteps is None else num_timesteps,
                text_temperature=config.sample.think_text_temperature,
                text_top_p=config.sample.think_text_top_p,
                text_top_k=config.sample.think_text_top_k,
            )
        else:
            if 'deterministic' not in kwargs:
                kwargs['deterministic'] = True
            gen_kwargs = dict(
                num_timesteps=config.sample.eval_num_steps if num_timesteps is None else num_timesteps,
                text_temperature=config.sample.eval_think_text_temperature,
                text_top_p=config.sample.eval_think_text_top_p,
                text_top_k=config.sample.eval_think_text_top_k,
            )

        inferencer = self.inferencer
        accelerator = self.accelerator
        model = inferencer.model
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            inferencer.model = unwrapped_model
            outputs = inferencer.batch_generate_images(
                input_lists=input_lists,
                image_shape=(config.resolution, config.resolution),
                cfg_text_scale=config.sample.cfg_text_scale,
                cfg_img_scale=config.sample.cfg_image_scale if task == 'edit' else 1.0,
                cfg_interval=config.sample.cfg_interval_edit if task == 'edit' else config.sample.cfg_interval,
                timestep_shift=config.sample.timestep_shift,
                cfg_renorm_min=config.sample.cfg_renorm_min,
                cfg_renorm_type=config.sample.cfg_renorm_type_edit if task == 'edit' else config.sample.cfg_renorm_type,
                think=think,
                **gen_kwargs,
                **kwargs,
            )
            inferencer.model = model

        if not return_log_probs:
            return outputs[0]

        images, latents, image_log_probs, think_texts, think_text_per_token_log_probs, contexts = outputs
        if isinstance(think_text_per_token_log_probs, tuple):
            think_text_per_token_log_probs = think_text_per_token_log_probs[0]
            print('text logprob', think_text_per_token_log_probs[0].dtype)
        return images, latents, image_log_probs, think_texts, think_text_per_token_log_probs, contexts

    def generate_text(self, input_lists, config, think=False,
                      task='ti2t', return_log_probs=False, **kwargs):
        """Text generation via batch_generate_texts.

        return_log_probs=False: returns (output_texts, think_texts).
        return_log_probs=True:  returns (output_texts, output_lp, think_texts, think_lp, contexts).
        """
        inferencer = self.inferencer
        accelerator = self.accelerator
        model = inferencer.model
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            inferencer.model = unwrapped_model
            outputs = inferencer.batch_generate_texts(
                input_lists=input_lists,
                return_log_probs=return_log_probs,
                return_middle_context=return_log_probs,
                think=think,
                text_temperature=config.sample.think_text_temperature,
                text_top_p=config.sample.think_text_top_p,
                text_top_k=config.sample.think_text_top_k,
                **kwargs,
            )
            inferencer.model = model

        if not return_log_probs:
            output_texts, think_texts = outputs
            return output_texts, think_texts

        output_texts, output_texts_per_token_log_probs, think_texts, think_texts_per_token_log_probs, contexts = outputs

        if isinstance(output_texts_per_token_log_probs, tuple):
            output_texts_per_token_log_probs = output_texts_per_token_log_probs[0]

        if isinstance(think_texts_per_token_log_probs, tuple):
            think_texts_per_token_log_probs = think_texts_per_token_log_probs[0]

        return output_texts, output_texts_per_token_log_probs, think_texts, think_texts_per_token_log_probs, contexts

    def compute_rewards(self, input_data, return_future=False):
        """Async reward computation. Moved from train.py compute_rewards()."""
        rewards_future = self.executor.submit(self.reward_fn, input_data)

        if return_future:
            return rewards_future

        rewards, reward_metadata = rewards_future.result()
        rewards = {
            key: torch.as_tensor(value, device=self.accelerator.device).float()
            for key, value in rewards.items()
        }
        return rewards, reward_metadata


# Need torch for compute_rewards
import torch

from . import t2i, ti2t, reflect, mixed, self_spectrareward_t2i, vrx
