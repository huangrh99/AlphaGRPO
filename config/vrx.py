# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fair BAGEL VR-X baselines for vanilla GRPO and UniVR / VR-GRPO."""

import json
import os
import imp

from ml_collections import config_dict

bagel = imp.load_source("bagel", os.path.join(os.path.dirname(__file__), "bagel.py"))


def _vrx_sources():
    raw = os.environ.get("VRX_TRAIN_SOURCES")
    if raw:
        return json.loads(raw)
    root = os.environ.get("VRX_DATA_ROOT", "")
    if not root:
        return {}
    names = {
        "agibot": "Agibot",
        "egodex": "EgoDex",
        "videocraft": "VideoCraftBench",
        "epic_kitchen": "Epic_Kitchen",
        "zebra_count": "Zebra_Count",
        "zebra_jigsaw": "Zebra_Jigsaw",
        "visual_search": "Visual_Search",
    }
    conventional = {
        name: os.path.join(root, directory) for name, directory in names.items()
    }
    if any(os.path.isdir(path) for path in conventional.values()):
        return {name: path for name, path in conventional.items() if os.path.isdir(path)}
    # The Hugging Face release is partitioned as RL-001, RL-002, ... and keeps
    # the logical source in each parquet row.
    return {"vrx_rl": root}


def _int_or_mapping(value, default=None):
    if not value:
        return default
    value = value.strip()
    return json.loads(value) if value.startswith("{") else int(value)


def _apply_trust_region_profile(config):
    profile = os.environ.get("VRX_TRUST_REGION_PROFILE", "alphagrpo_native")
    config.vrx.trust_region_profile = profile
    if profile == "alphagrpo_native":
        config.train.adv_clip_max = 5
        config.train.image_clip_range = 1e-5
        config.train.image_clip_range_high = 1e-5
        config.train.image_beta = 0.0
        config.train.recompute_log_prob = False
        config.train.advantage_std_ddof = 0
    elif profile == "univr_matched":
        # The asymmetric 0.2/0.3 token PPO interval maps cleanly to a ratio
        # interval. The continuous BAGEL KL below has the same role, not the
        # same numerical scale, as UniVR's token low-var KL.
        config.train.adv_clip_max = 1000000000
        config.train.image_clip_range = 0.2
        config.train.image_clip_range_high = 0.3
        config.train.image_beta = 1e-2
        config.train.recompute_log_prob = True
        config.train.advantage_std_ddof = 1
    else:
        raise ValueError(f"Unknown VRX_TRUST_REGION_PROFILE: {profile}")

    overrides = {
        "VRX_ADV_CLIP_MAX": ("adv_clip_max", float),
        "VRX_IMAGE_CLIP_RANGE": ("image_clip_range", float),
        "VRX_IMAGE_CLIP_RANGE_HIGH": ("image_clip_range_high", float),
        "VRX_IMAGE_BETA": ("image_beta", float),
        "VRX_ADVANTAGE_STD_DDOF": ("advantage_std_ddof", int),
    }
    for env_name, (field, parser) in overrides.items():
        if env_name in os.environ:
            setattr(config.train, field, parser(os.environ[env_name]))
    if "VRX_RECOMPUTE_LOG_PROB" in os.environ:
        config.train.recompute_log_prob = os.environ["VRX_RECOMPUTE_LOG_PROB"] == "1"


def _vrx_base():
    config = bagel._base_config()
    config.seed = int(os.environ.get("VRX_SEED", str(config.seed)))
    config.num_epochs = int(os.environ.get("VRX_NUM_EPOCHS", str(config.num_epochs)))
    config.save_freq = int(os.environ.get("VRX_SAVE_FREQ", str(config.save_freq)))
    config.model.bagel_model_path = os.environ.get(
        "BAGEL_MODEL_PATH", config.model.bagel_model_path
    )
    config.model.vae_path = os.environ.get(
        "BAGEL_VAE_PATH",
        os.path.join(config.model.bagel_model_path, "ae.safetensors"),
    )
    config.train.lora_path = os.environ.get("BAGEL_VRX_SFT_CHECKPOINT") or None
    config.train.task = "vrx"
    config.train.algorithm = "grpo"
    config.train.image_algorithm = None
    config.train.true_ratio = True
    config.train.rollout_correction = False
    config.train.async_reward_fn = True
    config.train.learning_rate = float(os.environ.get("VRX_LEARNING_RATE", "5e-5"))
    config.train.adam_weight_decay = float(
        os.environ.get("VRX_WEIGHT_DECAY", str(config.train.adam_weight_decay))
    )
    config.lora_rank = int(os.environ.get("VRX_LORA_RANK", "32"))
    config.lora_alpha = int(os.environ.get("VRX_LORA_ALPHA", str(2 * config.lora_rank)))

    group_size = int(os.environ.get("VRX_GROUP_SIZE", "6"))
    config.sample.num_image_per_prompt = group_size
    config.sample.train_batch_size = group_size
    config.sample.test_batch_size = 1
    config.sample.think = False
    config.sample.num_steps = int(os.environ.get("VRX_SDE_STEPS", "16"))
    config.sample.eval_num_steps = int(
        os.environ.get("VRX_EVAL_SDE_STEPS", str(config.sample.eval_num_steps))
    )
    config.train.num_train_timesteps = min(
        int(os.environ.get("VRX_TRAIN_TIMESTEPS", "10")), config.sample.num_steps
    )
    config.train.timestep_fraction = config.train.num_train_timesteps / config.sample.num_steps
    config.sample.sde_window_range = (0, config.train.num_train_timesteps)
    config.train.gradient_accumulation_steps = 1
    config.text_reward_fn = {}
    config.train.use_think_text_format_reward = False

    config.dataset = os.environ.get("VRX_DATA_ROOT", "")
    config.prompt_fn = "vrx"
    config.vrx = config_dict.ConfigDict()
    config.vrx.train_sources = _vrx_sources()
    val_sources = os.environ.get("VRX_VAL_SOURCES")
    config.vrx.val_sources = json.loads(val_sources) if val_sources else config.vrx.train_sources
    config.vrx.max_samples_per_source = _int_or_mapping(
        os.environ.get("VRX_MAX_SAMPLES_PER_SOURCE")
    )
    config.vrx.max_val_samples_per_source = _int_or_mapping(
        os.environ.get("VRX_MAX_VAL_SAMPLES_PER_SOURCE"), 8
    )
    config.vrx.num_reasoning_steps = int(os.environ.get("VRX_REASONING_STEPS", "4"))
    config.vrx.use_gt_step_count = os.environ.get("VRX_USE_GT_STEP_COUNT", "0") == "1"
    config.vrx.use_instruction_step_count = os.environ.get("VRX_USE_INSTRUCTION_STEP_COUNT", "0") == "1"
    config.vrx.max_reasoning_steps = int(os.environ.get("VRX_MAX_REASONING_STEPS", "8"))
    config.vrx.require_matching_gt_steps = (
        os.environ.get("VRX_REQUIRE_MATCHING_GT_STEPS", "1") == "1"
    )
    config.vrx.required_gt_frames = (
        config.vrx.num_reasoning_steps
        if config.vrx.require_matching_gt_steps
        and not config.vrx.use_gt_step_count
        and not config.vrx.use_instruction_step_count
        else None
    )
    image_size = os.environ.get("VRX_IMAGE_SIZE")
    config.vrx.image_size = int(image_size) if image_size else None
    # Accelerator needs the number of backward calls before model construction.
    # The default fixed-step strategy therefore accumulates all image transitions
    # in one visual trajectory before a real optimizer update.
    config.train.backward_calls_per_sample = (
        config.vrx.num_reasoning_steps * config.train.num_train_timesteps
    )

    config.vrx.reward_mode = "released_code"
    config.vrx.format_weight = float(os.environ.get("VRX_FORMAT_WEIGHT", "0.2"))
    # Released Pref-GRPO normalizes the base weights, then inserts a raw
    # win-rate weight of 1.0 and normalizes again. 2.1 reproduces the resulting
    # effective ratio format:win-rate:sampled-frame:count = 2:21:10:2.
    config.vrx.global_weight = float(os.environ.get("VRX_GLOBAL_WEIGHT", "2.1"))
    config.vrx.local_weight = float(os.environ.get("VRX_LOCAL_WEIGHT", "1.0"))
    config.vrx.image_count_weight = float(os.environ.get("VRX_IMAGE_COUNT_WEIGHT", "0.2"))
    config.vrx.reason_lambda = float(os.environ.get("VRX_REASON_LAMBDA", "2.0"))
    config.vrx.sampled_frames_ratio = float(
        os.environ.get("VRX_SAMPLED_FRAMES_RATIO", "0.5")
    )

    config.num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
    config.total_batch_size = group_size * config.num_gpus
    config.logdir = os.environ.get("VRX_LOGDIR", "./logs/vrx")
    config.save_dir = config.logdir
    _apply_trust_region_profile(config)
    return config


def _reward_spec(config, mode, pairwise, enable_local):
    return {
        "weight": 1.0,
        "kwargs": {
            "mode": mode,
            "pairwise": pairwise,
            "mock": os.environ.get("UNIVR_REWARD_MOCK", "0") == "1",
            "api_url": os.environ.get("UNIVR_REWARD_API_URL") or None,
            "model": os.environ.get("UNIVR_REWARD_MODEL") or None,
            "api_key": os.environ.get("UNIVR_REWARD_API_KEY") or None,
            "timeout": float(os.environ.get("UNIVR_REWARD_TIMEOUT", "200")),
            "pairwise_timeout": float(
                os.environ.get("UNIVR_PAIRWISE_TIMEOUT", "2000")
            ),
            "max_vlm_workers": int(
                os.environ.get("UNIVR_MAX_VLM_WORKERS", "128")
            ),
            "format_weight": 0.0 if mode == "vanilla" else config.vrx.format_weight,
            "global_weight": config.vrx.global_weight,
            "local_weight": config.vrx.local_weight,
            "image_count_weight": 0.0 if mode == "vanilla" else config.vrx.image_count_weight,
            "reason_lambda": config.vrx.reason_lambda,
            "sampled_frames_ratio": config.vrx.sampled_frames_ratio,
            "seed": config.seed,
            "enable_local": enable_local,
        },
    }


def _validation_reward_spec():
    return {
        "weight": 1.0,
        "kwargs": {
            "mock": os.environ.get("UNIVR_REWARD_MOCK", "0") == "1",
            "api_url": os.environ.get("UNIVR_REWARD_API_URL") or None,
            "model": os.environ.get("UNIVR_REWARD_MODEL") or None,
            "api_key": os.environ.get("UNIVR_REWARD_API_KEY") or None,
            "timeout": float(os.environ.get("UNIVR_VAL_TIMEOUT", "60")),
            "max_vlm_workers": int(
                os.environ.get("UNIVR_VAL_MAX_VLM_WORKERS", "32")
            ),
        },
    }


def vrx_sft_initialized_vanilla_grpo():
    config = _vrx_base()
    config.reward_fn = {
        "univr_vrgrpo": _reward_spec(config, mode="vanilla", pairwise=False, enable_local=False)
    }
    config.val_reward_fn = {
        "univr_vrgrpo_validation": _validation_reward_spec()
    }
    config.logdir = os.environ.get("VRX_LOGDIR", "./logs/vrx_vanilla_grpo")
    config.save_dir = config.logdir
    return config


def vrx_sft_initialized_univr_grpo():
    config = _vrx_base()
    config.reward_fn = {
        "univr_vrgrpo": _reward_spec(
            config, mode="released_code", pairwise=True, enable_local=True
        )
    }
    config.val_reward_fn = {
        "univr_vrgrpo_validation": _validation_reward_spec()
    }
    config.logdir = os.environ.get("VRX_LOGDIR", "./logs/vrx_univr_grpo")
    config.save_dir = config.logdir
    return config


def vrx_univr_smoke():
    config = vrx_sft_initialized_univr_grpo()
    config.sanity_check = True
    config.num_epochs = 1
    config.sample.num_image_per_prompt = 2
    config.sample.train_batch_size = 2
    config.sample.test_batch_size = 1
    config.sample.num_steps = 2
    config.train.num_train_timesteps = 1
    config.train.timestep_fraction = 0.5
    config.sample.sde_window_range = (0, 1)
    smoke_reasoning_steps = int(os.environ.get("VRX_SMOKE_REASONING_STEPS", "2"))
    config.vrx.num_reasoning_steps = smoke_reasoning_steps
    config.vrx.max_reasoning_steps = smoke_reasoning_steps
    config.vrx.required_gt_frames = (
        smoke_reasoning_steps if config.vrx.require_matching_gt_steps else None
    )
    config.train.backward_calls_per_sample = smoke_reasoning_steps
    if config.vrx.train_sources:
        first_source = next(iter(config.vrx.train_sources))
        config.vrx.train_sources = {
            first_source: config.vrx.train_sources[first_source]
        }
        config.vrx.val_sources = config.vrx.train_sources
    smoke_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    smoke_samples = int(os.environ.get(
        "VRX_SMOKE_MAX_SAMPLES_PER_SOURCE", str(smoke_world_size)
    ))
    config.vrx.max_samples_per_source = smoke_samples
    config.vrx.max_val_samples_per_source = smoke_samples
    smoke_image_size = os.environ.get("VRX_SMOKE_IMAGE_SIZE", "512")
    config.vrx.image_size = (
        None if smoke_image_size.lower() in {"", "0", "none", "native"}
        else int(smoke_image_size)
    )
    config.num_gpus = smoke_world_size
    config.total_batch_size = 2 * smoke_world_size
    config.train.gradient_accumulation_steps = 1
    config.train.dataloader_num_workers = 0
    config.reward_fn["univr_vrgrpo"]["kwargs"]["mock"] = True
    config.val_reward_fn["univr_vrgrpo_validation"]["kwargs"]["mock"] = True
    config.save_freq = 1
    config.eval_freq = 1000000
    config.log_image_freq = 1
    config.logdir = os.environ.get("VRX_LOGDIR", "./logs/vrx_univr_smoke")
    config.save_dir = config.logdir
    return config


def get_config(name):
    return globals()[name]()

