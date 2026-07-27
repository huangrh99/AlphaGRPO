<div align="center">

# AlphaGRPO
### Self-Reflective Multimodal Generation via Decompositional Verifiable Reward

[![AlphaGRPO arXiv](https://img.shields.io/badge/AlphaGRPO-arXiv%3A2605.12495-b31b1b.svg)](https://arxiv.org/abs/2605.12495)
[![SpectraReward arXiv](https://img.shields.io/badge/SpectraReward-arXiv%3A2607.11886-b31b1b.svg)](https://arxiv.org/abs/2607.11886)
[![Project Page](https://img.shields.io/badge/AlphaGRPO-Page-blue)](https://huangrh99.github.io/AlphaGRPO)
[![Project Page](https://img.shields.io/badge/SpectraReward-Page-blue)](https://huangrh99.github.io/SpectraReward/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-AlphaGRPO_Models-blue.svg)](https://huggingface.co/collections/huangrh9/alphagrpo)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-SpectraReward_Models-blue.svg)](https://huggingface.co/collections/huangrh9/spectrareward)
[![Papers with Code: #2 on WISE](https://paperswithcode.co/api/v1/papers/2607.11886/leaderboard-badge.svg?eval=14485&live=1)](https://paperswithcode.co/benchmark/wise?task=image-understanding&eval=14485)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
</div>

This is the official repo for **AlphaGRPO: Self-Reflective Multimodal Generation via Decompositional Verifiable Reward**.

**TL;DR:** AlphaGRPO enables multimodal generation RL training across text and image generation for AR-Diffusion-native unified multimodal models, such as [BAGEL](https://github.com/bytedance-seed/BAGEL). It supports training on reasoning text-to-image generation and self-reflective refinement.

This codebase flexibly supports different RL methods for image and text generation: **FlowGRPO**, **DiffusionNFT**, and **AWM** for images; **GRPO** for text.

<p align="center">
  <img src="assets/figures/png/teaser.png" width="92%" />
</p>

## 📣 News

- **[2026/07/14]** We release **SpectraReward**, *[Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation](https://arxiv.org/abs/2607.11886)*. SpectraReward turns a frozen pretrained MLLM into a training-free, off-the-shelf reward model for text-to-image RL by measuring how well the prompt can be "read back" from the generated image. Code and configs are in this repo; see the [SpectraReward](#spectrareward) section and [`docs/SPECTRAREWARD.md`](docs/SPECTRAREWARD.md).
- **[2026/06/12]** We release **AlphaGRPO**, an RL framework for multimodal generation training on [BAGEL](https://github.com/bytedance-seed/BAGEL). Supporting tasks include reasoning text-to-image generation and self-reflective refinement.
- **[2026/05/13]** We released the paper, "AlphaGRPO: Self-Reflective Multimodal Generation via Decompositional Verifiable Reward" on [arXiv](https://arxiv.org/abs/2605.12495).

## 🏗️ Overview

AlphaGRPO trains unified multimodal models with a decompositional reward design: complex prompts are broken into verifiable semantic and quality checks, and text/image generation steps can be optimized with separate RL algorithms.

The framework is step-list driven, so each task can explicitly define its own text and image generation sequence. This supports simple reasoning text-to-image generation as well as multi-stage self-reflective refinement without hardcoding a fixed generation pattern.

<p align="center">
  <img src="assets/figures/png/method.png" width="92%" />
</p>

## 🔧 Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Git LFS

The bundled `alphagrpo20k` JSONL dataset files are tracked with Git LFS. Install Git LFS before cloning, or pull LFS files after cloning:

```bash
git lfs install
git lfs pull
```

If the dataset looks unusually small, check that `alpha_grpo/dataset/alphagrpo20k/train.jsonl` is not a Git LFS pointer file.

## 🚀 Quick Start

### AlphaGRPO (DVReward)

Download the base [BAGEL](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) model weights before training.

Then update `huggingface_models_root` in `config/bagel.py` to point to `/path/to/huggingface_models/`. The bundled DVReward dataset is under `alpha_grpo/dataset/alphagrpo20k/`; to use your own data, see [`docs/DVREWARD.md`](docs/DVREWARD.md).

Deploy the reward server in a separate terminal on each reward-server node:

```bash
ip=YOUR_IP
port=YOUR_PORT
bash scripts/serve_reward_model.sh $ip $port
```

Single-node training example:

```bash
ip=YOUR_REWARD_SERVER_IP
port=YOUR_REWARD_SERVER_PORT

export PYTHONPATH=$PYTHONPATH:$(pwd)/Bagel/
# IPv6:
export OPENAI_API_URL=http://[${ip}]:${port}/v1
# IPv4:
# export OPENAI_API_URL=http://${ip}:${port}/v1
export OPENAI_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct

task=alphagrpo_reflect  # alphagrpo_t2iThink is also available
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
  alpha_grpo/train.py --config config/bagel.py:${task}
```

Multi-node training example:

```bash
ip=YOUR_REWARD_SERVER_IP
port=YOUR_REWARD_SERVER_PORT

export PYTHONPATH=$PYTHONPATH:$(pwd)/Bagel/
export OPENAI_API_URL=http://[${ip}]:${port}/v1  # IPv6 reward-server address
export OPENAI_MODEL_NAME=Qwen/Qwen3-VL-30B-A3B-Instruct
    
# We use 8 nodes × 8 A100: 1 GPU/node for serving reward model, 7 for training.
# Scale gradient_accumulation_steps inversely with GPU count to maintain batch size.
task=alphagrpo_reflect # for self-reflective refinement task
# task=alphagrpo_t2iThink  # for reasoning text-to-image task
torchrun --nnodes=$NUM_NODES --node_rank=$RANK --nproc_per_node=7 \
  alpha_grpo/train.py --config config/bagel.py:${task}
```

### SpectraReward

*[Read It Back: Pretrained MLLMs Are Zero-Shot Reward Models for Text-to-Image Generation](https://arxiv.org/abs/2607.11886)*

**SpectraReward** turns a frozen pretrained MLLM into a training-free reward model for text-to-image RL. It scores how well the prompt can be *read back* from the generated image, using the **mean image-conditioned prompt log-likelihood** as the reward. **Self-SpectraReward** is the unified-model special case, where BAGEL's own understanding branch scores its generation branch, with no external reward model. See [`docs/SPECTRAREWARD.md`](docs/SPECTRAREWARD.md) for the full method, training setup, backbone support, and results.

**External SpectraReward.** Run the reward MLLM on a remote server so it does not share GPU memory with BAGEL:

```bash
# On the reward-server node(s): serve the reward MLLM (8-way data-parallel here).
bash scripts/serve_spectrareward.sh Qwen/Qwen3-VL-30B-A3B-Instruct 0.0.0.0 18090 8
```

```bash
# On the training side: point to the server and launch as usual.
export PYTHONPATH=$PYTHONPATH:$(pwd)/Bagel/
export SPECTRAREWARD_MODEL_ID=Qwen/Qwen3-VL-30B-A3B-Instruct
export SPECTRAREWARD_URL=http://<reward_server_ip>:18090   # IPv6: http://[${ip}]:18090

torchrun --nnodes=$NUM_NODES --node_rank=$RANK --nproc_per_node=8 \
  alpha_grpo/train.py --config config/bagel.py:spectrareward_t2i_awm
```

**Self-SpectraReward.** No external reward model is needed; BAGEL scores itself:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/Bagel/

torchrun --nnodes=$NUM_NODES --node_rank=$RANK --nproc_per_node=8 \
  alpha_grpo/train.py --config config/bagel.py:self_spectrareward_t2i_awm
```

For the in-process fallback, compatible MLLM backbones, configs, and experimental results, see [`docs/SPECTRAREWARD.md`](docs/SPECTRAREWARD.md).

## 📊 Evaluation

Install eval dependencies:
```bash
bash Bagel/eval_install.sh
```

Run benchmarks:
```bash
cd Bagel
# Evaluate on downstream benchmarks (GenEval, TIIF, WISE, DPG, GEdit)
task=geneval # dpg, tiif, wise, gedit

# Evaluate multiple LoRA checkpoints. Configure `jobname` and `checkpoint_list` in the corresponding script before running.
bash scripts/eval/run_${task}_multickpt.sh

# Evaluate one LoRA checkpoint directly.
export BAGEL_LORA_PATH=/path/to/checkpoint/hf_model/
bash scripts/eval/run_${task}.sh

# Conduct self-reflective refinement on downstream tasks.
task=geneval # dpg, tiif
bash scripts/eval/run_${task}_multickpt_reflect.sh
```

## 📊 Main Results

<p align="center">
  <img src="assets/figures/png/main_results_t2i.png" width="85%" />
</p>

> **Text-to-image benchmarks.** RT2I denotes the AlphaGRPO variant trained on the reasoning text-to-image task. Inf. SRR denotes inference-time self-reflective refinement.

<p align="center">
  <img src="assets/figures/png/main_results_gedit.png" width="65%" />
</p>

> **GEdit-Bench-EN.** Editing transfer performance: AlphaGRPO improves GEdit scores without training on editing tasks.

## 📦 Train on Your Own Dataset

To train AlphaGRPO on a custom prompt set with our **Decompositional Verifiable Reward (DVReward)**, see [`DVREWARD.md`](docs/DVREWARD.md) for the prompt-decomposition pipeline and how to wire the resulting dataset into training.

## 🧩 Extending AlphaGRPO

Text and image steps support independent RL algorithms (e.g., GRPO for text + AWM or DiffusionNFT for image). **New tasks, algorithms, and rewards can be added modularly.** See [docs/extending.md](docs/extending.md) for details.

## 👍 Acknowledgements

This project builds upon [BAGEL](https://github.com/bytedance-seed/BAGEL) and [Flow-GRPO](https://github.com/yifan123/flow_grpo). We thank the authors for their excellent work.



## ✏️ Citation

If you find AlphaGRPO or SpectraReward useful to your research, please consider citing:
```bibtex
@inproceedings{huang2026alphagrpo,
  title={AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in Unified Multimodal Models via Decompositional Verifiable Reward},
  author={Huang, Runhui and Wu, Jie and Yang, Rui and Liu, Zhe and Zhao, Hengshuang},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}

@misc{huang2026readitback,
  title={Read It Back: Pretrained {MLLMs} Are Zero-Shot Reward Models for Text-to-Image Generation},
  author={Huang, Runhui and Zhang, Qihui and Liu, Zhe and Gao, Yu and Wu, Jie and Zhao, Hengshuang},
  year={2026},
  eprint={2607.11886},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.11886}
}
```

## 🔒 License

This project is released under the [Apache License 2.0](LICENSE).
