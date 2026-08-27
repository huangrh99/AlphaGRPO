# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""
AlphaGRPO training entrypoint.

This script wires together:
- Config loading via --config (default config/bagel_grpo.py)
- Model/reward construction (Bagel/Qwen2/SigLIP, rewards, EMA, LoRA)
- Rollout -> reward -> GRPO losses -> optimization loop
- Optional evaluation and checkpointing
- Logging vi1andb/tensorboard

Notes:
- Ignore context grad.
- Save the cache.
- Optimized for speed.
"""

from math import e
import sys
import re
import os
import datetime
import contextlib
import time
import json
from collections import defaultdict
from concurrent import futures
from copy import deepcopy
from functools import partial
import random

import yaml
import numpy as np
import tqdm
from PIL import Image

import torch
import torch.distributed as dist

from absl import app, flags
from ml_collections import config_flags

from accelerate import Accelerator, DeepSpeedPlugin, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import (
    set_seed,
    ProjectConfiguration,
    DataLoaderConfiguration,
    FullyShardedDataParallelPlugin,
    gather,
    gather_object,
)
from accelerate.logging import get_logger

import wandb
from transformers import get_scheduler

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from peft.utils import get_peft_model_state_dict

# Third-party non-standard modules, then local package imports
from stat_tracking import PerPromptStatTracker
from rewards.rewards import multi_score
from safetensors.torch import load_file
from trl.models import create_reference_model
from rewards.ema import EMAModuleWrapper
from utils import *
from dataset import build_dataloader
import algorithms
import tasks

# import bagel relative modules
from inferencer import InterleaveInferencer
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# ====== Config / Logging ======
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/bagel.py", "Training configuration.")

logger = get_logger(__name__)


# ====== Model / Transform Loading ======
def load_bagel_weight(model, model_path, device, dtype):

    env_load_bagel_weight_path = os.environ.get('BAGEL_MODEL_WEIGHT', None)
    if env_load_bagel_weight_path is not None:
        state_dict = load_file(env_load_bagel_weight_path, device='cpu')
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict,assign=True,)
        print(f'Bagel model loaded with dtype: {dtype} from {env_load_bagel_weight_path}')
    else:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map={"": device},  # Load to specified device
            offload_buffers=True,
            dtype=dtype,  # Use the specified dtype for consistency
            force_hooks=True,
            offload_folder="/tmp/offload"
        )
        print(f'Bagel model loaded with dtype: {dtype} from {model_path}')
    model = model.to(device).eval()
    return model


def load_bagel_model(args, device='cuda', dtype=torch.bfloat16):
    """Load Bagel model for training"""
    # Load model configuration
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model.bagel_model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model.bagel_model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model.bagel_model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.model.max_latent_size,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model.bagel_model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model = load_bagel_weight(model, args.model.bagel_model_path, device, dtype)
    model.language_model.resize_token_embeddings(len(tokenizer))

    vae_transform = ImageTransform(args.model.vae_max_image_size, args.model.vae_min_image_size, 16)
    vit_transform = ImageTransform(args.model.vit_max_image_size, args.model.vit_min_image_size, 14)

    return model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform


# ====== Logprob Utilities (training-phase recomputation) ======
def compute_image_logprob(inferencer, model, gen_step, timestep_index, config, dtype=torch.bfloat16,
                     gen_context=None, cfg_text_context=None, cfg_img_context=None,
                     task='edit', algorithm='grpo',
                     **kwargs):
    """
    Compute log probability of a transition for the Bagel model.
    gen_step: GenerationStep with latents/next_latents data.
    """
    image_shape = gen_step.image_shape or (config.resolution, config.resolution)
    if algorithm in ('awm', 'nft'):
        clean_latents = gen_step.next_latents[:, -1]  # the final clean latents

        # Lazy cache: first call at this timestep generates noise + caches sigma,
        # subsequent calls (ref/ema/training) reuse for consistency.
        j = timestep_index - gen_step.sde_window[0]
        ts_count = gen_step.sde_window[1] - gen_step.sde_window[0]
        if gen_step.fm_noise is None:
            gen_step.fm_noise = [None] * ts_count
            gen_step.fm_sigma = [None] * ts_count
        if gen_step.fm_noise[j] is None:
            gen_step.fm_noise[j] = torch.randn_like(clean_latents)

        image_log_probs, prev_latents_mean, std_dev_t, model_output = inferencer.compute_image_logprob_fm(
            clean_latents, timestep_index,
            gen_context=gen_context,
            image_shape=image_shape,
            num_timesteps=config.sample.num_steps,
            timestep_shift=config.sample.timestep_shift,
            model=model,
            noise=gen_step.fm_noise[j],
            **kwargs,
        )
        timesteps = inferencer.get_timesteps(
                config.sample.num_steps, timestep_shift=config.sample.timestep_shift, device=image_log_probs.device)
        timestep = timesteps[timestep_index]
        gen_step.fm_sigma[j] = timestep  # cache sigma for NFT loss
        image_log_probs = -(torch.pow(-image_log_probs + 1e-10, config.train.ghuber_power) - torch.pow(torch.tensor(1e-10, device=image_log_probs.device, dtype=image_log_probs.dtype), config.train.ghuber_power)) * timestep.view(-1) / config.train.ghuber_power
    else:
        latents = gen_step.latents[:, timestep_index]
        next_latents = gen_step.next_latents[:, timestep_index]

        image_log_probs, prev_latents_mean, std_dev_t, model_output = inferencer.compute_image_logprob(
            latents, next_latents, timestep_index,
            gen_context=gen_context, cfg_text_context=cfg_text_context, cfg_img_context=cfg_img_context,
            image_shape=image_shape,
            num_timesteps=config.sample.num_steps,
            timestep_shift=config.sample.timestep_shift,
            cfg_text_scale=config.sample.cfg_text_scale,
            cfg_img_scale=config.sample.cfg_image_scale if task=='edit' else 1.0,
            noise_level=config.sample.noise_level,
            cfg_interval=config.sample.cfg_interval_edit if task=='edit' else config.sample.cfg_interval,
            cfg_renorm_min=config.sample.cfg_renorm_min,
            cfg_renorm_type=config.sample.cfg_renorm_type_edit if task=='edit' else config.sample.cfg_renorm_type,
            cfg_type='parallel',
            use_flowcps=config.sample.use_flowcps,
            model=model,
            **kwargs,
        )
    return image_log_probs.float(), prev_latents_mean, std_dev_t, model_output


def compute_text_logprob(inferencer, model, think_texts, gen_context, config, compute_entropy=False, dtype=torch.bfloat16):
    """
    Compute log probability of a transition for the Bagel model.
    """
    text_per_token_log_probs, text_token_lens, text_entropies, gen_context = inferencer.compute_text_logprob(
        think_texts, gen_context,
        temperature=config.sample.think_text_temperature,
        return_context=True, compute_entropy=compute_entropy,
        model=model,
    )
    return text_per_token_log_probs.float(), text_token_lens, text_entropies, gen_context


# ====== Entry Point ======
def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = True

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    config.run_name = os.path.basename(os.path.normpath(config.logdir)) + '_' + unique_id

    print('config.run_name', config.run_name)
    timing_profiler = TimingProfiler()

    if config.train.image_algorithm is None:
        config.train.image_algorithm = config.train.algorithm

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")

            resume_from_ckpt = sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1]

            print(f"Auto resume from {resume_from_ckpt}")
            config.resume_from = os.path.join(
                config.resume_from,
                resume_from_ckpt,
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    deepspeed_plugin = None
    if hasattr(config.train, "deepspeed_config") and config.train.deepspeed_config:
        with open(config.train.deepspeed_config, "r") as f:
            deepspeed_config_dict = json.load(f)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config_dict)

    # Gradient accumulation = total backward() calls per optimizer step.
    # For standard step pattern [text, image]:
    #   - text step: 1 backward call
    #   - image step: num_train_timesteps backward calls
    # Formula must be set statically before Accelerator init.
    # For non-standard patterns (multi-round), use config.train.backward_calls_per_sample override.
    num_train_timesteps = max(int(config.sample.num_steps * config.train.timestep_fraction), 1) if not config.train.disable_image_grpo else 1
    actual_gradient_accumulation_steps = config.train.gradient_accumulation_steps * num_train_timesteps
     # if use think so there an additional update,
    if not config.train.disable_image_grpo and config.sample.think:
        actual_gradient_accumulation_steps += config.train.gradient_accumulation_steps

    if config.train.mix_t2i_task:
        # two samples in one rollout.
        actual_gradient_accumulation_steps *= 2

    # Optional override for non-standard step patterns
    if config.train.backward_calls_per_sample is not None:
        actual_gradient_accumulation_steps = (
            config.train.gradient_accumulation_steps * config.train.backward_calls_per_sample)
        if config.train.mix_t2i_task:
            actual_gradient_accumulation_steps *= 2

    accelerator = Accelerator(
        log_with=config.logger_type,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=actual_gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )

    if is_deepspeed_enabled(accelerator):
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"] = config.sample.train_batch_size

    if accelerator.is_main_process:
        # Clean up save_dir and log_dir before training starts
        if not config.resume_from:
            # Only clean up if not resuming from checkpoint
            import subprocess

            # Clean up save_dir
            save_dir = os.path.expanduser(config.save_dir)
            if os.path.exists(save_dir):
                logger.info(f"Cleaning up save_dir: {save_dir}")
            #     # shutil.rmtree(save_dir)
            #     print(f'Existing experiment directory, please manually clean the ${save_dir}')
            #     exit(0)
                # subprocess.run(['rm', '-rf', save_dir], check=False)

            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Created clean save_dir: {save_dir}")

            # Clean up log_dir
            log_dir = os.path.expanduser(config.logdir)
            if os.path.exists(log_dir):
                logger.info(f"Cleaning up log_dir: {log_dir}")
            #     # shutil.rmtree(log_dir)
            #     print(f'Existing experiment directory, please manually clean the ${log_dir}')
            #     exit(0)
                # subprocess.run(['rm', '-rf', log_dir], check=False)

            os.makedirs(log_dir, exist_ok=True)
            logger.info(f"Created clean log_dir: {log_dir}")

        if config.logger_type == "wandb":
            # Enhanced wandb configuration for better offline logging
            wandb_init_kwargs = {
                "name": config.run_name,
                "dir": os.path.join(config.logdir, config.run_name),  # Specify wandb directory
                "save_code": True,  # Save code snapshot
                "resume": "allow",  # Allow resuming from checkpoints
            }

            # Create wandb directory if it doesn't exist
            os.makedirs(wandb_init_kwargs["dir"], exist_ok=True)

            # Log some system information
            logger.info(f"Wandb logs will be stored in: {wandb_init_kwargs['dir']}")
            logger.info(f"To sync later, run: wandb sync {wandb_init_kwargs['dir']}")

            wandb.init(
                project=config.wandb.project,
                **wandb_init_kwargs
            )

            wandb.config.update(config.to_dict())
            wandb.config.update(dict(commit_id=os.popen('git rev-parse HEAD').read().strip()))
            wandb.config.update(dict(world_size=accelerator.num_processes))
        else:  # tensorboard.
            accelerator.init_trackers(
                project_name=config.wandb.project,
                config=sanitize_config(config.to_dict()),
            )

    logger.info(f"\n{config}")

    cuda_device = accelerator.device
    if cuda_device.type == "cuda" and cuda_device.index is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(cuda_device)
    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed + accelerator.process_index, device_specific=True)

    # Log DeepSpeed information
    if is_deepspeed_enabled(accelerator):
        deepspeed_info = get_deepspeed_info(accelerator)
        logger.info(f"{deepspeed_info}")

    # Warn about ZeRO Stage 3 special handling
    if is_deepspeed_zero_stage_3(accelerator):
        logger.info("DeepSpeed ZeRO Stage 3 detected - using modified gradient accumulation handling")
        logger.info(
            f"Read plugin config: gradient accumulation {accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps']}")

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    logger.info(f"Inference dtype: {inference_dtype}")

    # load scheduler, tokenizer and models.
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform = load_bagel_model(
        config, str(accelerator.device), dtype=inference_dtype)

    if config.train.use_liger_kernel:
        apply_liger_kernel_to_qwen2_navit(
            rope=True,
            swiglu=False,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
            rms_norm=True,
            model=model.language_model,
        )
        print('use liger kernal')

    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    inferencer.compute_dtype = inference_dtype
    vae_dtype = inference_dtype if is_deepspeed_enabled(accelerator) else torch.float32
    inferencer.vae_model.to(accelerator.device, dtype=vae_dtype).eval()

    # make the progress bar nicer
    inferencer.model.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    if config.use_lora:
        inferencer.model.to(accelerator.device)

        # all linear in attention and mlp of LLM.
        # target_modules = r"language_model\.model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen)|mlp\.(gate_proj|up_proj|down_proj)|mlp_moe_gen\.(gate_proj|up_proj|down_proj))"

        # All linear layers in attention and MLP of LLM, including MoE generation variants.
        target_modules = r"language_model\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj(_moe_gen)?|mlp(_moe_gen)?\.(gate_proj|up_proj|down_proj))"

        transformer_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        if config.train.lora_path:
            inferencer.model = PeftModel.from_pretrained(inferencer.model, config.train.lora_path)
            # 使用PeftModel.from_pretrained load后所有参数的requires_grad都是False，需要set_adapter来使得adapter参数梯度为True
            inferencer.model.set_adapter("default")
        else:
            inferencer.model = get_peft_model(inferencer.model, transformer_lora_config)

    inferencer.model.to(inference_dtype)
    if config.train.use_gradient_checkpointing:
        if is_deepspeed_enabled(accelerator):
            inferencer.model.language_model.gradient_checkpointing_enable()

    model = inferencer.model
    model_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    if config.train.image_beta > 0 or config.train.text_beta > 0:
        if config.use_lora:
            ref_model = model
        else:
            ref_model = create_reference_model(model)
            ref_model = ref_model.cpu()
    else:
        ref_model = None

    # KL EMA: separate LoRA adapter for adaptive KL regularization
    if config.train.ema_beta > 0 and config.use_lora:
        model.add_adapter("kl_ema", transformer_lora_config)
        model.set_adapter("kl_ema")
        kl_ema_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        model.set_adapter("default")
        # Initialize kl_ema adapter with current training adapter weights
        for src_param, tgt_param in zip(
            model_trainable_parameters, kl_ema_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param
    else:
        kl_ema_parameters = None

    if config.train.ema:
        # 平均影响到之前的20*8=160个step
        ema = EMAModuleWrapper(model_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    else:
        ema = None

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if config.train.und_learning_rate is not None:
        und_param_pattern = r'self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)'

        und_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if re.search(und_param_pattern, name):
                    und_params.append(param)
                else:
                    other_params.append(param)

        optimizer_grouped_parameters = [
            {"params": und_params, "lr": config.train.und_learning_rate},
            {"params": other_params, "lr": config.train.learning_rate},
        ]

        optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )
    else:
        optimizer = optimizer_cls(
            model_trainable_parameters,
            lr=config.train.learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            weight_decay=config.train.adam_weight_decay,
            eps=config.train.adam_epsilon,
        )

    # prepare prompt and reward fn
    reward_fn = multi_score(accelerator.device, config.reward_fn) if config.reward_fn else None
    val_reward_fn = multi_score(
        accelerator.device, getattr(config, "val_reward_fn", {})
    ) if getattr(config, "val_reward_fn", None) else None
    text_reward_fn = multi_score(accelerator.device, config.text_reward_fn) if getattr(config, 'text_reward_fn', None) else None

    train_dataloader, test_dataloader = build_dataloader(config, accelerator)

    # Create learning rate scheduler
    total_steps = config.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=config.train.warmup_steps,
        num_training_steps=total_steps,
    )

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    stat_tracker = None
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.sample.global_std, ddof=getattr(config.train, "advantage_std_ddof", 0)
        )

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    # ignore train dataloader.
    model, optimizer, test_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, test_dataloader, lr_scheduler)

    if accelerator.is_main_process:
        print(model)

    # IMPORTANT: Update inferencer.model to use the wrapped model
    # This ensures that inference and training use the same model instance
    inferencer.model = model

    # executor to perform callbacks asynchronously.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # ====== Initialize Task and Algorithm Dispatchers ======
    TaskClass = tasks.get_task(config.train.task)
    task = TaskClass(
        inferencer=inferencer,
        accelerator=accelerator,
        executor=executor,
        reward_fn=reward_fn,
        text_reward_fn=text_reward_fn,
        val_reward_fn=val_reward_fn,
    )
    text_algo = algorithms.get_algorithm(config.train.algorithm)
    image_algo = algorithms.get_algorithm(config.train.image_algorithm)

    # Train!
    total_steps = config.num_epochs * len(train_dataloader)

    # Sanity check mode
    sanity_check = getattr(config, 'sanity_check', False)
    if sanity_check:
        config.save_freq = 2
        config.eval_freq = 2
        logger.info("Enable sanity check mode")

    logger.info("***** Running Online Training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Batches per epoch = {len(train_dataloader)}")
    logger.info(f"  Total steps = {total_steps}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")

    total_params = sum(p.numel() for p in model_trainable_parameters)
    logger.info(f"  Total number of trainable parameters = {total_params}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        logger.info(f"Successfully loaded checkpoint from {config.resume_from}")

        # For DeepSpeed, additional logging
        if is_deepspeed_enabled(accelerator):
            logger.info("DeepSpeed checkpoint loaded successfully")

        current_step = accelerator.step
        global_step = current_step // actual_gradient_accumulation_steps
        batches_to_skip = global_step * config.train.gradient_accumulation_steps
        start_epoch = batches_to_skip // len(train_dataloader)
        print(batches_to_skip, len(train_dataloader))
        if batches_to_skip > 0:
            print(f"Resuming training: skipping first {batches_to_skip} batches.")
            train_dataloader = accelerator.skip_first_batches(train_dataloader, batches_to_skip)
    else:
        global_step = 0
        start_epoch = 0

    if config.train.eval_before_train:
        task.eval(test_dataloader, config, 0, autocast)

    for epoch in range(start_epoch, config.num_epochs):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
            train_dataloader.batch_sampler.set_epoch(epoch)

        info = defaultdict(list)

        # collect all samples for training.
        all_samples = []

        rollout_progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}: online training",
            disable=not accelerator.is_local_main_process,
            position=0,
        )

        def should_change_to_train_mode(step):
            return step % (config.train.gradient_accumulation_steps * config.sample.num_batches_per_ppo_epoch) == 0

        rollout_step = 0  # the number of rollout samples.
        for step_index, batch_data in enumerate(rollout_progress_bar):
            rollout_progress_bar.set_description(f"Epoch {epoch} | global_step {global_step} rollouting")

            #################### SAMPLING ####################
            model.eval()

            with timing_profiler('rollout_samples'):
                sample, should_skip = task.rollout(
                    config=config,
                    tokenizer=tokenizer, batch_data=batch_data,
                    global_step=global_step, stat_tracker=stat_tracker,
                )

            if should_skip:
                continue

            all_samples.extend(sample) if isinstance(sample, list) else all_samples.append(sample)
            rollout_step += 1

            # continue rollout or to train.
            if not should_change_to_train_mode(rollout_step):
                continue

            #################### TRAINING ####################
            log_image_freq = config.log_image_freq

            model.train()

            # transfer for training. Clean up the cache.
            train_samples = all_samples
            all_samples = []

            # Need recompute: (1) repeat text (2) ppo_epoch > 1 (3) ppo_batch > 1 (4) algorithm is 'awm'
            if config.train.recompute_log_prob:
                for sample_index, sample in enumerate(train_samples):
                    with timing_profiler('recompute_log_prob'):
                        gen_context, cfg_text_context, cfg_img_context = deepcopy(sample['contexts'])

                        with autocast():
                            with torch.no_grad():
                                for step_idx, gen_step in enumerate(sample['generation_steps']):
                                    is_last = (step_idx == len(sample['generation_steps']) - 1)

                                    # Input conditioning steps: advance context only, no loss
                                    if gen_step.step_type == 'input_text':
                                        cfg_text_context = deepcopy(gen_context)
                                        gen_context = inferencer.update_context_text(gen_step.texts, gen_context)
                                        cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)
                                        continue
                                    elif gen_step.step_type == 'input_image':
                                        gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                        cfg_text_context = deepcopy(gen_context)
                                        continue

                                    # Generation steps: compute loss
                                    if gen_step.step_type in ('think_text', 'text'):
                                        old_lp, _, _, gen_context = compute_text_logprob(
                                            inferencer, model, gen_step.texts, gen_context, config)
                                        gen_step.rollout_text_per_token_log_probs = gen_step.text_per_token_log_probs
                                        gen_step.text_per_token_log_probs = old_lp.detach()

                                    elif gen_step.step_type == 'image':
                                        ts_list = list(range(gen_step.sde_window[0], gen_step.sde_window[1]))
                                        task_type = gen_step.task_type or 't2i'
                                        all_lp, all_mo = [], []
                                        for j, ts in enumerate(ts_list):
                                            lp, _, _, mo = compute_image_logprob(
                                                inferencer, model, gen_step, ts, config, inference_dtype,
                                                gen_context=deepcopy(gen_context),
                                                cfg_text_context=deepcopy(cfg_text_context),
                                                cfg_img_context=deepcopy(cfg_img_context),
                                                task='t2i' if task_type == 't2i' else 'edit',
                                                algorithm=config.train.image_algorithm,
                                            )
                                            all_lp.append(lp.detach())
                                            all_mo.append(mo)
                                        gen_step.rollout_image_log_probs = gen_step.image_log_probs
                                        gen_step.image_log_probs = torch.stack(all_lp, dim=1)
                                        gen_step.image_outputs = torch.stack(all_mo, dim=1)

                                    # Context advancement between steps
                                    if not is_last and gen_step.step_type == 'image' and gen_step.images:
                                        gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                        cfg_text_context = deepcopy(gen_context)
                                    if not is_last and gen_step.step_type == 'text':
                                        cfg_text_context = inferencer.update_context_text(gen_step.texts, cfg_text_context)
                                        cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)

            # compute kl
            if config.train.image_beta > 0 or config.train.text_beta > 0:
                for sample_index, sample in enumerate(train_samples):
                    with timing_profiler('compute_kl'):
                        gen_context, cfg_text_context, cfg_img_context = deepcopy(sample['contexts'])

                        with autocast():
                            with torch.no_grad():
                                ref_model_context = ref_model.disable_adapter if config.use_lora else contextlib.nullcontext
                                with ref_model_context():
                                    for step_idx, gen_step in enumerate(sample['generation_steps']):
                                        is_last = (step_idx == len(sample['generation_steps']) - 1)

                                        # Input conditioning steps: advance context only
                                        if gen_step.step_type == 'input_text':
                                            cfg_text_context = deepcopy(gen_context)
                                            gen_context = inferencer.update_context_text(gen_step.texts, gen_context)
                                            cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)
                                            continue
                                        elif gen_step.step_type == 'input_image':
                                            gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                            cfg_text_context = deepcopy(gen_context)
                                            continue

                                        # Generation steps
                                        if gen_step.step_type in ('think_text', 'text'):
                                            # Always run text step to advance gen_context,
                                            # even if text_beta == 0 (image KL needs correct context).
                                            ref_lp, _, _, gen_context = compute_text_logprob(
                                                inferencer, ref_model, gen_step.texts, gen_context, config)
                                            gen_step.ref_text_per_token_log_probs = ref_lp.detach()

                                        elif gen_step.step_type == 'image':
                                            ts_list = list(range(gen_step.sde_window[0], gen_step.sde_window[1]))
                                            task_type = gen_step.task_type or 't2i'
                                            all_ref_means, all_ref_mo = [], []
                                            for j, ts in enumerate(ts_list):
                                                _, ref_mean, _, ref_mo = compute_image_logprob(
                                                    inferencer, ref_model, gen_step, ts, config, inference_dtype,
                                                    gen_context=deepcopy(gen_context),
                                                    cfg_text_context=deepcopy(cfg_text_context),
                                                    cfg_img_context=deepcopy(cfg_img_context),
                                                    task='t2i' if task_type == 't2i' else 'edit',
                                                    algorithm=config.train.image_algorithm,
                                                )
                                                all_ref_means.append(ref_mean.detach())
                                                all_ref_mo.append(ref_mo.detach())
                                            gen_step.ref_prev_latents_means = torch.stack(all_ref_means, dim=1)
                                            gen_step.ref_model_output = torch.stack(all_ref_mo, dim=1)

                                        # Context advancement between steps
                                        if not is_last and gen_step.step_type == 'image' and gen_step.images:
                                            gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                            cfg_text_context = deepcopy(gen_context)
                                        if not is_last and gen_step.step_type == 'text':
                                            cfg_text_context = inferencer.update_context_text(gen_step.texts, cfg_text_context)
                                            cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)

            # compute ema kl (separate LoRA adapter for adaptive KL)
            if config.train.ema_beta > 0 and kl_ema_parameters is not None:
                for sample_index, sample in enumerate(train_samples):
                    with timing_profiler('compute_ema_kl'):
                        gen_context, cfg_text_context, cfg_img_context = deepcopy(sample['contexts'])

                        with autocast():
                            with torch.no_grad():
                                model.set_adapter("kl_ema")
                                for step_idx, gen_step in enumerate(sample['generation_steps']):
                                    is_last = (step_idx == len(sample['generation_steps']) - 1)

                                    # Input conditioning steps: advance context only
                                    if gen_step.step_type == 'input_text':
                                        cfg_text_context = deepcopy(gen_context)
                                        gen_context = inferencer.update_context_text(gen_step.texts, gen_context)
                                        cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)
                                        continue
                                    elif gen_step.step_type == 'input_image':
                                        gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                        cfg_text_context = deepcopy(gen_context)
                                        continue

                                    # Generation steps
                                    if gen_step.step_type in ('think_text', 'text'):
                                        _, _, _, gen_context = compute_text_logprob(
                                            inferencer, model, gen_step.texts, gen_context, config)

                                    elif gen_step.step_type == 'image':
                                        ts_list = list(range(gen_step.sde_window[0], gen_step.sde_window[1]))
                                        task_type = gen_step.task_type or 't2i'
                                        all_ema_mo = []
                                        for j, ts in enumerate(ts_list):
                                            _, _, _, ema_mo = compute_image_logprob(
                                                inferencer, model, gen_step, ts, config, inference_dtype,
                                                gen_context=deepcopy(gen_context),
                                                cfg_text_context=deepcopy(cfg_text_context),
                                                cfg_img_context=deepcopy(cfg_img_context),
                                                task='t2i' if task_type == 't2i' else 'edit',
                                                algorithm=config.train.image_algorithm,
                                            )
                                            all_ema_mo.append(ema_mo.detach())
                                        gen_step.ema_model_output = torch.stack(all_ema_mo, dim=1)

                                    # Context advancement between steps
                                    if not is_last and gen_step.step_type == 'image' and gen_step.images:
                                        gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                        cfg_text_context = deepcopy(gen_context)
                                    if not is_last and gen_step.step_type == 'text':
                                        cfg_text_context = inferencer.update_context_text(gen_step.texts, cfg_text_context)
                                        cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)
                                model.set_adapter("default")

            if accelerator.is_local_main_process:
                print('==========================')
                for si, s in enumerate(train_samples):
                    steps_desc = [
                        f"{gs.step_type}({gs.sde_window})" if gs.step_type == 'image' else gs.step_type
                        for gs in s.get('generation_steps', [])
                    ]
                    print(global_step, si, steps_desc, s.get('task'))
                print('==========================')

            train_progress_bar = tqdm(
                total=config.train.ppo_epoch * len(train_samples),
                desc=f"Epoch {epoch}: online training",
                disable=not accelerator.is_local_main_process,
                position=0,
            )

            # Train for multiple iterations per sample
            for iter_step in range(config.train.ppo_epoch):
                # Currently, it only shuffle the batch-level data not sample-level data.

                if config.train.shuffle_train_sample:
                    random.shuffle(train_samples)

                for sample_index, sample in enumerate(train_samples):
                    train_progress_bar.set_description(f"Epoch {epoch} | global_step {global_step} | iter_step {iter_step} | training_sample {sample_index}")
                    train_progress_bar.update(1)

                    # sync reward if neeeded.
                    if config.train.async_reward_fn:
                        if not sample['is_postprocessed']:
                            with timing_profiler('wait_async_reward_fn_finish'):
                                sample, should_skip = sample["reward_postprocess_fn"](config, sample, stat_tracker=stat_tracker)
                                if should_skip:
                                    rollout_step -= 1  # Ensure the next rollout round requires only the skipped number of samples.
                                    logger.info(f"Skip data of prompt {np.unique(sample['prompts'])} in global_step {global_step}")
                                    continue

                    # Record reward metrics (task-specific)
                    task.collect_metrics(sample, info)

                    gen_context, cfg_text_context, cfg_img_context = deepcopy(sample['contexts'])

                    if accelerator.is_local_main_process:
                        steps_desc = [
                            f"{gs.step_type}({gs.sde_window})" if gs.step_type == 'image' else gs.step_type
                            for gs in sample.get('generation_steps', [])
                        ]
                        print(global_step, iter_step, sample_index, steps_desc, sample.get('task'))

                    for step_idx, gen_step in enumerate(sample['generation_steps']):
                        is_last = (step_idx == len(sample['generation_steps']) - 1)

                        # Input conditioning steps: advance context only, no loss
                        if gen_step.step_type == 'input_text':
                            with torch.no_grad():
                                cfg_text_context = deepcopy(gen_context)
                                gen_context = inferencer.update_context_text(gen_step.texts, gen_context)
                                cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)
                            continue
                        elif gen_step.step_type == 'input_image':
                            with torch.no_grad():
                                gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                cfg_text_context = deepcopy(gen_context)
                            continue

                        # Generation steps: compute loss
                        if gen_step.step_type in ('think_text', 'text'):
                            timing_profiler.start('text_train_step')
                            with accelerator.accumulate(model):
                                text_per_token_log_probs, text_token_lens, text_entropies, gen_context = compute_text_logprob(
                                        inferencer, model, gen_step.texts, gen_context, config, compute_entropy=True)

                                info, step_loss = text_algo.compute_text_loss(
                                    config, sample, accelerator, info,
                                    gen_step=gen_step,
                                    text_per_token_log_probs=text_per_token_log_probs,
                                    text_entropies=text_entropies,
                                    text_token_lens=text_token_lens,
                                    get_high_entropy_mask=get_high_entropy_mask,
                                )

                                # text loss scale: balance with subsequent image timesteps
                                remaining_img_ts = sum(
                                    len(range(s.sde_window[0], s.sde_window[1]))
                                    for s in sample['generation_steps'][step_idx+1:]
                                    if s.step_type == 'image' and s.sde_window
                                )
                                accelerator.backward(step_loss * max(1, remaining_img_ts) * config.train.text_loss_weight)

                                if accelerator.sync_gradients:
                                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad()
                            timing_profiler.end('text_train_step')

                        elif gen_step.step_type == 'image':
                            task_type = gen_step.task_type or 't2i'

                            timing_profiler.start('image_train_step')
                            ts_list = list(range(gen_step.sde_window[0], gen_step.sde_window[1]))
                            cache_g, cache_ct, cache_ci = gen_context, cfg_text_context, cfg_img_context

                            for j, ts in enumerate(ts_list):
                                with accelerator.accumulate(model):
                                    g, ct, ci = deepcopy(cache_g), deepcopy(cache_ct), deepcopy(cache_ci)

                                    with autocast():
                                        img_lp, prev_mean, std_t, mo = compute_image_logprob(
                                            inferencer, model, gen_step, ts, config, inference_dtype,
                                            gen_context=g, cfg_text_context=ct, cfg_img_context=ci,
                                            task='t2i' if task_type == 't2i' else 'edit',
                                            algorithm=config.train.image_algorithm,
                                        )

                                    info, step_loss = image_algo.compute_image_loss(
                                        config, sample, accelerator, info,
                                        gen_step=gen_step, j=j,
                                        image_log_probs=img_lp, prev_latents_mean=prev_mean,
                                        std_dev_t=std_t, model_output=mo,
                                    )

                                    accelerator.backward(step_loss)

                                    if accelerator.sync_gradients:
                                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

                                    optimizer.step()
                                    lr_scheduler.step()
                                    optimizer.zero_grad()
                            timing_profiler.end('image_train_step')

                            # Context advancement if not last step
                            if not is_last and gen_step.images:
                                with torch.no_grad():
                                    gen_context = inferencer.update_context_image(gen_step.images, gen_context)
                                    cfg_text_context = deepcopy(gen_context)

                        # Context advancement for output text between steps
                        if not is_last and gen_step.step_type == 'text':
                            with torch.no_grad():
                                cfg_text_context = inferencer.update_context_text(gen_step.texts, cfg_text_context)
                                cfg_img_context = inferencer.update_context_text(gen_step.texts, cfg_img_context)

                    # Per-sample debug prints (after all steps)
                    if accelerator.is_local_main_process:
                        for key in ['image_policy_loss', 'text_policy_loss', 'image_loss', 'text_loss']:
                            if len(info.get(key, [])):
                                print(key, info[key][-1])
                        if len(info.get('text_ratio', [])):
                            print('text_ratio', info['text_ratio'][-1])
                        if len(info.get('image_ratio', [])):
                            print('image_ratio', info['image_ratio'][-1])

                    if accelerator.sync_gradients:
                        if ema is not None:
                            ema.step(model_trainable_parameters, global_step)

                        # Update KL EMA adapter weights via decay
                        if kl_ema_parameters is not None:
                            with torch.no_grad():
                                if config.train.kl_ema_decay_type == 'constant':
                                    kl_decay = config.train.kl_ema_decay
                                elif config.train.kl_ema_decay_type == 'linear':
                                    kl_decay = min(config.train.kl_ema_decay, 0.001 * global_step)
                                else:
                                    raise ValueError(f"Unknown kl_ema_decay_type: {config.train.kl_ema_decay_type}")
                                for src_param, tgt_param in zip(
                                    model_trainable_parameters, kl_ema_parameters, strict=True
                                ):
                                    tgt_param.data.copy_(tgt_param.data * kl_decay + src_param.detach().data * (1.0 - kl_decay))

                        if global_step % log_image_freq == 0:
                            task.log_training_step(train_samples, accelerator, global_step, config)

                        if (global_step + 1) % config.eval_freq == 0:
                            with timing_profiler('eval'):
                                task.eval(test_dataloader, config, global_step, autocast)

                        # Periodically save checkpoint
                        if (global_step + 1) % config.save_freq == 0:
                            save_ckpt(config.save_dir, model, global_step, accelerator, ema, config)

                        info_reduced = {k: torch.mean(torch.stack(v)).cuda() for k, v in info.items() if len(v)}
                        info_reduced.update(timing_profiler.pop_records())
                        info_reduced = accelerator.reduce(info_reduced, reduction="mean")

                        # this three don't need to reduce.
                        info_reduced.update({"grad_norm": grad_norm, "epoch": epoch, "step": step_index})

                        if accelerator.is_main_process:
                            if config.logger_type == 'wandb':
                                wandb.log(info_reduced, step=global_step)
                            else:
                                accelerator.log(info_reduced, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

    # Final evaluation and save
    task.eval(test_dataloader, config, global_step, autocast)
    save_ckpt(config.save_dir, model, global_step, accelerator, ema, config)

    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)