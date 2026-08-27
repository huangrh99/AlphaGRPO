# BAGEL VR-X: Vanilla GRPO 与 UniVR / VR-GRPO

本文档以本地 `UniVR/UniVR_RL` released code 为准，记录源码对齐结论、BAGEL 适配边界、当前机器路径、运行前必须修改的配置和启动命令。

## 结论

迁移目标已经按正确边界实现：保留 AlphaGRPO 与 BAGEL 连续 latent/SDE actor，只迁移 VR-X 数据、纯图像多步轨迹、UniVR 的组内成对全局奖励、组共享采样帧奖励、格式与 GT 帧数门控，以及 released-code 奖励聚合。

不能称为逐 token 完全复现。UniVR 官方 actor 是 Emu3.5 离散视觉 token PPO；当前 actor 是 BAGEL 的连续转移 `z_t -> z_(t-1)`。Emu VQ tokenizer、vLLM patch、HybridEngine、verl trainer 均未迁移，这与任务要求一致。

正式实验目前还有两个外部前置条件未满足：

1. 本机没有找到 BAGEL VR-X SFT LoRA adapter，必须设置 `BAGEL_VRX_SFT_CHECKPOINT`。
2. 真实 Qwen3-VL 奖励服务尚未验证，必须设置并检查 `UNIVR_REWARD_API_URL`；mock 只能验证管线，不能验证奖励质量。

## 源码对齐表

| 项目 | UniVR released code | AlphaGRPO VR-X | 状态与说明 |
|---|---|---|---|
| Backbone | Emu3.5 | BAGEL-7B-MoT | 有意不同；任务明确要求保留 BAGEL |
| 动作空间 | 离散视觉 token | 连续 latent/SDE transition | 不可数值等价，保留各自原生 log-prob |
| 纯图像 CoT | 一个 response 中解析多张图 | 多个 `GenerationStep(step_type=image)`，上一帧进入下一步 context | 方法语义对齐，无中间文本 CoT |
| group 建立 | 每个输入先生成 UUID，再 interleave repeat `rollout.n` 次 | stable uid 加本地 K-repeat sampler | 功能对齐；stable uid 额外满足跨 rollout 可识别要求 |
| group 所在设备 | verl 在全局 batch 上按 uid 计算 | 完整 G 条候选留在同一 rank，随后全局 gather advantage | 对 AlphaGRPO 的进程内 group-aware reward 是必要适配 |
| group size | YAML 为 8，官方 LoRA 启动脚本覆盖为 6 | `VRX_GROUP_SIZE`，默认 6 | 对齐可配置 |
| 全局奖励 | uid 内所有有效候选两两比较，win rate，TIE 记 0.5 | 同样枚举所有 unordered pairs | 对齐 |
| 单候选组 | 有效候选得 0.5 | 0.5 | 对齐 |
| 局部奖励 | 按 GT 长度随机采样 `max(1, int(T*0.5))`，同 uid 共用 indices | 同分布、同组共享；使用 seed 加 uid 的确定性 RNG | 统计语义对齐；具体随机序列不是逐调用 bitwise 对齐 |
| 帧数错误 | 生成帧数不等于 GT 帧数时 sampled-frame 为 0，最后 overall 强制为 0 | 已改为严格比较生成轨迹与 GT 轨迹长度 | 对齐 |
| format | Emu image-token 语法、有效图块和尺寸一致 | PIL 有效、非空、计划帧数、尺寸一致 | BAGEL 结构等价适配，不检查 Emu token |
| reward mode | released code 默认 Pref-GRPO | `released_code` | 对齐 |
| README critical step | README 描述 CLIP variance | released `emu3.py` 实际没有该 selector | 未虚构实现；`paper` mode 明确报错 |
| HPSv3 / DINOv2 | 代码存在，默认关闭 | 未迁移 | 与官方实际默认一致 |
| validation reward | `emu3_val.py` 单样本 absolute VLM；format 只作为 gate，最终 overall 直接取 VLM | 独立 `univr_vrgrpo_validation` scorer 与 `VRXTask.eval` | 对齐；不套用训练 image-count gate |
| advantage mean | uid 内均值 | stable uid 内均值 | 对齐 |
| advantage std | `torch.std`，默认 sample std，epsilon 为 `1e-6` | profile 可选 ddof；native 为 ddof 0，matched 为 ddof 1，当前 epsilon 为 `1e-4` | matched 的 ddof 对齐；epsilon 仍有很小差异，文档明确保留 |
| advantage clip | 官方 GRPO advantage 不裁剪 | BAGEL native 为正负 5；matched 近似关闭 | profile 可控 |
| old-policy log-prob | rollout 后重算 token `old_log_probs` | rollout SDE log-prob；matched 开启重算 | 方法角色对齐，动作概率定义不同 |
| ratio | `exp(log_pi - log_pi_old)` | 同形式，作用于 image transition | 形式对齐 |
| PPO clip | lower 0.2，upper 0.3，negative-adv dual clip 3 | native 为 1e-5 对称；matched 为 0.2/0.3，无 dual clip | 只部分对齐；不能把 token dual clip 冒充连续 actor 的精确等价物 |
| clipfrac | 分别记录 high 和 low dual-clip | 记录超出 BAGEL lower/upper interval 的比例 | 语义已修正，但指标定义不完全相同 |
| reference KL | token low-var KL，coef `1e-2` | reference/current transition mean 距离除以 `2*sigma_t^2`，matched beta `1e-2` | 正则角色相同，数学形式与尺度不同 |
| loss aggregation | response token masked mean | 每个 image transition timestep mean | 保留动作空间原生聚合 |
| credit assignment | 一个 trajectory advantage 广播到 response tokens | 一个 trajectory advantage用于全部视觉步和 SDE timestep | outcome-supervised 语义对齐 |
| rollout correction | 官方脚本未启用额外 correction | VR-X 配置为 false | 对齐 |
| LoRA RL | rank 64，alpha 128，target `all-linear` | 默认 rank 32，alpha 64，BAGEL LLM attention 和 MLP regex | 未数值对齐；可用环境变量改为 64，但 target 和参数量仍不等价 |
| LoRA SFT | 官方脚本 rank 64，alpha 128，target q/k/v/o | 必须加载同一个 BAGEL VR-X SFT adapter 供两条 baseline 共用 | adapter 自身 rank 由其 `adapter_config.json` 决定 |
| optimizer | lr `1e-6`，weight decay `1e-2`，grad norm 1 | 默认 lr `5e-5`，weight decay `1e-4`，grad norm 1 | 未默认对齐；下面给出运行时覆盖，两个 baseline 必须保持相同 |
| epochs / save | 10 epochs，save 50 | 10 epochs，save 20 | 训练预算需在正式实验配置中统一 |

核心回答：UniVR 没有使用一个需要整体移植的全新 GRPO estimator。算法层的主要创新在 trajectory reward construction；AlphaGRPO 的 BAGEL `GRPOAlgorithm` 可以复用。离散 token actor loss、token KL 和 dual clip 不应直接替换 BAGEL image-GRPO。

## Reward 精确语义

官方 YAML 设置 `format=.2, vlm=.7`，函数默认还启用 `sampled_frames=1.0`、`image_count=.2` 和 `pref_grpo_winrate=1.0`。released code 先对基础权重归一化，进入 Pref-GRPO 后再加入 raw win-rate weight 做第二次归一化。最终有效系数为：

```text
format       =  2 / 35
win rate     = 21 / 35
sampled      = 10 / 35
image count  =  2 / 35
```

AlphaGRPO 用等效的未归一化权重 `.2, 2.1, 1.0, .2`，得到完全相同的最终系数。令 `R_g` 为 win rate，`R_s` 为组共享采样帧均分：

```text
R_reason = R_g - 2 * abs(R_g - R_s)
R = (0.2 * R_format + 3.1 * R_reason + 0.2 * R_count) / 3.5
```

format 为 0 或生成帧数不等于 GT 帧数时，overall 为 0。局部帧 VLM 与官方一致解析 `TASK_SCORE`、`VISUAL_SCORE` 后取均值；pairwise 严格解析官方 `PREFER: A/B/TIE`。网络错误返回中性 0.5 或 TIE，无法解析的 absolute/frame score 返回 0.3，与 released code 的容错方向一致。

VLM 调用已按官方源码对齐：三套 rubric 直接取自 `emu3.py`，只发送一个 `user` message；图片顺序为 GT、input/reference、generated，pairwise 为 GT、reference、A、B；PNG、temperature 0.1、max tokens 2048、global/frame timeout 200 秒、pairwise timeout 2000 秒。absolute、所有 pair 和所有 sampled frame 分阶段通过线程池并发。AlphaGRPO 使用直接 HTTP POST，官方使用 OpenAI Python client，但发送到 `/v1/chat/completions` 的 JSON 协议一致。并发作用域仍有框架差异：官方是 Ray reward worker 内全局并发，AlphaGRPO 是每个训练 rank 的完整本地 group 内并发。
validation 另行使用 `emu3_val.py` 的任务分类和 task-specific rubric（Agibot、EgoDex、VideoCraft），不会复用训练期的通用 trajectory rubric。

## 数据对齐与本机路径

本机数据：

```text
/research/cvl-zhourzh8/VR-X-SFT-RL/VR-X-RL
3601 rows, 80 parquet files, about 1.1 GB
目录结构: RL-001 到 RL-007
GT 帧数范围: 1 到 8
```

本地 parquet 的主要来源计数为：EGODEX 989、AgiBot alpha/beta 合计 1080、Zebra-CoT 893、Visual Search 485、EPIC 103、VideoCraft 51。官方 YAML 的训练目标配比是 agibot 500、egodex 300、epic 103、videocraft 51、zebra_count 500、zebra_jigsaw 200、visual_search 200。

当前 reader 会：

- 自动识别官方命名目录，或把本机 `RL-*` 根目录作为一个物理 partition 递归读取。
- 保留 parquet 原始 `dataset_source`，另存 `dataset_partition`，不再覆盖真实来源。
- 解码 problem/answer bytes 为 PIL RGB，保留全部非图像 metadata。
- 生成稳定 uid。
- 支持 `VRX_MAX_SAMPLES_PER_SOURCE` 为整数或 JSON mapping。
- 固定 T 时默认过滤到 `len(GT)==T`，防止任务天生长度不匹配导致全 0 reward。

注意：本机 HF 分区不等同于官方七个物理目录，所以只设置一个 JSON 上限不能逐字重现官方 `_apply_per_source_sampling` 的七源 oversampling。若论文实验要求精确数据配比，应先按 parquet 的 `dataset_source` 和 Zebra `task_name` 建立七个 source mapping 或离线 manifest；当前默认是正确读取完整 HF release，不应宣称精确复现官方 1854 条 source mix。

## 运行前配置


配置入口集中在 `config/vrx.py`，不要修改 `config/bagel.py` 的集群占位路径。正式运行至少要检查下面这些变量：

| 类别 | 环境变量 | 当前默认 | UniVR 官方值或用途 |
|---|---|---|---|
| 模型 | `BAGEL_MODEL_PATH` | BAGEL 占位路径 | 必须设为本机 BAGEL-7B-MoT |
| VAE | `BAGEL_VAE_PATH` | 模型目录下 ae.safetensors | 通常无需单独改 |
| SFT 初始化 | `BAGEL_VRX_SFT_CHECKPOINT` | 空 | 两条 baseline 必须使用同一 BAGEL VR-X SFT adapter |
| 数据根目录 | `VRX_DATA_ROOT` | 空 | 本机 VR-X RL 根目录 |
| source mapping | `VRX_TRAIN_SOURCES` / `VRX_VAL_SOURCES` | 自动识别 / 复用 train | JSON mapping；独立 validation 时必须修改 |
| 样本上限 | `VRX_MAX_SAMPLES_PER_SOURCE` / `VRX_MAX_VAL_SAMPLES_PER_SOURCE` | 无 / 8 | 整数或 JSON mapping |
| reward 服务 | `UNIVR_REWARD_API_URL` | 空 | OpenAI-compatible `/v1` 或完整 endpoint |
| reward model | `UNIVR_REWARD_MODEL` | `qwen3vl` | 必须与服务端 model id 完全一致 |
| reward key | `UNIVR_REWARD_API_KEY` | 空 | 服务需要鉴权时必填 |
| reward 模式 | `UNIVR_REWARD_MOCK` | 0 | 正式训练必须保持 0；仅 plumbing 测试可设 1 |
| train 并发 | `UNIVR_MAX_VLM_WORKERS` | 128 | 官方 LoRA 脚本为 128，YAML 为 256 |
| train timeout | `UNIVR_REWARD_TIMEOUT` / `UNIVR_PAIRWISE_TIMEOUT` | 200 / 2000 秒 | 分别对齐 global-frame / pairwise |
| val 并发与 timeout | `UNIVR_VAL_MAX_VLM_WORKERS` / `UNIVR_VAL_TIMEOUT` | 32 / 60 秒 | 对齐官方 validation |
| group size G | `VRX_GROUP_SIZE` | 6 | 对齐官方 LoRA 启动脚本 |
| visual steps T | `VRX_REASONING_STEPS` / `VRX_MAX_REASONING_STEPS` | 4 / 8 | 固定 T 时建议设为同一个值 |
| 动态步数 | `VRX_USE_GT_STEP_COUNT` / `VRX_USE_INSTRUCTION_STEP_COUNT` | 0 / 0 | 多卡 DDP 当前保持关闭 |
| GT 长度过滤 | `VRX_REQUIRE_MATCHING_GT_STEPS` | 1 | 固定 T 时只训练 `len(GT)==T` |
| SDE 预算 | `VRX_SDE_STEPS` / `VRX_TRAIN_TIMESTEPS` | 16 / 10 | rollout 与进入 loss 的 timestep 数 |
| validation SDE | `VRX_EVAL_SDE_STEPS` | 40 | deterministic validation 生成预算 |
| 分辨率 | `VRX_IMAGE_SIZE` | native | 512 只是显存选项 |
| LoRA | `VRX_LORA_RANK` / `VRX_LORA_ALPHA` | 32 / 64 | 官方 RL 为 64 / 128；target 仍不同 |
| optimizer | `VRX_LEARNING_RATE` / `VRX_WEIGHT_DECAY` | 5e-5 / 1e-4 | 官方为 1e-6 / 1e-2 |
| trust region | `VRX_TRUST_REGION_PROFILE` | `alphagrpo_native` | 可设 `univr_matched` |
| reward weights | `VRX_FORMAT_WEIGHT` / `VRX_GLOBAL_WEIGHT` / `VRX_LOCAL_WEIGHT` / `VRX_IMAGE_COUNT_WEIGHT` | .2 / 2.1 / 1 / .2 | 2.1 复现官方双重归一化 |
| reward 组合 | `VRX_REASON_LAMBDA` / `VRX_SAMPLED_FRAMES_RATIO` | 2 / .5 | 对齐 released code |
| 预算与输出 | `VRX_SEED` / `VRX_NUM_EPOCHS` / `VRX_SAVE_FREQ` / `VRX_LOGDIR` | base / 10 / 20 / logs | 官方 save 50 时覆盖；baseline 分目录 |

`univr_matched` profile 设置 sample std、0.2/0.3 ratio interval、BAGEL reference KL beta `1e-2` 和 old-log-prob recompute。逐项覆盖入口为：

```bash
export VRX_ADV_CLIP_MAX=1000000000
export VRX_IMAGE_CLIP_RANGE=0.2
export VRX_IMAGE_CLIP_RANGE_HIGH=0.3
export VRX_IMAGE_BETA=1e-2
export VRX_ADVANTAGE_STD_DDOF=1
export VRX_RECOMPUTE_LOG_PROB=1
```

这些只是连续 BAGEL actor 中有数学意义的映射；UniVR token dual-clip 3 没有被伪装成完全等价的 BAGEL 参数。

先进入仓库并使用现有 uv 环境：

```bash
cd /research/cvl-zhourzh8/AlphaGRPO
export UV=/research/cvl-zhourzh8/miniconda3/bin/uv
$UV pip install --python .venv/bin/python -r requirements.txt
```

本机已有路径：

```bash
export BAGEL_MODEL_PATH=/research/cvl-zhourzh8/models/BAGEL-7B-MoT
export BAGEL_VAE_PATH=/research/cvl-zhourzh8/models/BAGEL-7B-MoT/ae.safetensors
export VRX_DATA_ROOT=/research/cvl-zhourzh8/VR-X-SFT-RL/VR-X-RL
```

必须自行补上的地址：

```bash
export BAGEL_VRX_SFT_CHECKPOINT=/absolute/path/to/bagel-vrx-sft-lora
export UNIVR_REWARD_API_URL=http://reward-host:port/v1
export UNIVR_REWARD_MODEL=qwen3vl
export UNIVR_REWARD_API_KEY=your-key
```

`UNIVR_REWARD_API_URL` 可以是 `http://host:port/v1` 或完整的 `.../v1/chat/completions`，client 会自动补齐后者。奖励模型必须部署在训练 GPU 之外，不要把 30B VLM 加载到 BAGEL 训练卡。

共同实验参数建议显式设置，不要依赖默认值。`config/vrx.py` 还支持 `VRX_LORA_ALPHA`、`VRX_TRAIN_TIMESTEPS`、`VRX_SEED`、`VRX_NUM_EPOCHS`、`VRX_SAVE_FREQ`、四个 reward weight、`VRX_REASON_LAMBDA`、`VRX_SAMPLED_FRAMES_RATIO` 和 trust-region 单项覆盖：

```bash
export VRX_GROUP_SIZE=6
export VRX_REASONING_STEPS=4
export VRX_MAX_REASONING_STEPS=4
export VRX_REQUIRE_MATCHING_GT_STEPS=1
export VRX_SDE_STEPS=16
export VRX_EVAL_SDE_STEPS=40
export VRX_IMAGE_SIZE=512
export VRX_LORA_RANK=32
export VRX_LEARNING_RATE=5e-5
export VRX_WEIGHT_DECAY=1e-4
export VRX_TRUST_REGION_PROFILE=alphagrpo_native
export UNIVR_MAX_VLM_WORKERS=128
export UNIVR_REWARD_TIMEOUT=200
export UNIVR_PAIRWISE_TIMEOUT=2000
export UNIVR_VAL_MAX_VLM_WORKERS=32
export UNIVR_VAL_TIMEOUT=60
export VRX_LOGDIR=/research/cvl-zhourzh8/AlphaGRPO/logs/vrx_run
```

其中 `VRX_IMAGE_SIZE=512` 不是 UniVR 或 VR-X 的硬要求，只是显存控制项。删除它即保留 source resolution。固定 T 的多卡训练推荐保持 `VRX_REQUIRE_MATCHING_GT_STEPS=1`。`VRX_USE_GT_STEP_COUNT=1` 和 instruction-derived T 会让不同 rank 的 backward 次数不同，目前不推荐用于多卡 DDP。

若要数值采用官方 RL 的 rank 与 optimizer 参数，可覆盖：

```bash
export VRX_LORA_RANK=64
export VRX_LEARNING_RATE=1e-6
export VRX_WEIGHT_DECAY=1e-2
export VRX_TRUST_REGION_PROFILE=univr_matched
```

`VRX_WEIGHT_DECAY=1e-2` 对齐官方 weight decay。即使 rank 都是 64，Emu `all-linear` 与 BAGEL target regex 的 trainable parameter count 也不同。更重要的是，加载现有 `BAGEL_VRX_SFT_CHECKPOINT` 时实际 adapter rank 以 checkpoint 的 `adapter_config.json` 为准；Vanilla 与 UniVR 两条运行必须使用同一个 adapter。

## 如何运行

先做不使用 GPU 的结构测试：

```bash
cd /research/cvl-zhourzh8/AlphaGRPO
.venv/bin/python -m unittest discover -s tests -p test_vrx.py -v
```

四卡 3 到 6 的 UniVR / VR-GRPO 正式入口：

```bash
cd /research/cvl-zhourzh8/AlphaGRPO
CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=$PWD/Bagel \
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  alpha_grpo/train.py --config config/vrx.py:vrx_sft_initialized_univr_grpo
```

Vanilla GRPO 只替换 config 名并使用不同 logdir，其他环境变量必须完全相同：

```bash
export VRX_LOGDIR=/research/cvl-zhourzh8/AlphaGRPO/logs/vrx_vanilla
CUDA_VISIBLE_DEVICES=3,4,5,6 PYTHONPATH=$PWD/Bagel \
.venv/bin/torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  alpha_grpo/train.py --config config/vrx.py:vrx_sft_initialized_vanilla_grpo
```

UniVR run 应使用另一个目录：

```bash
export VRX_LOGDIR=/research/cvl-zhourzh8/AlphaGRPO/logs/vrx_univr
```

不能在真实训练中设置 `UNIVR_REWARD_MOCK=1`。mock reward 是由生成图像 hash 得到的确定性测试分，不是固定常数，但没有语义质量，不能作为实验结果。

显存不足时优先按相同规则同时修改两条 baseline：

```bash
export VRX_GROUP_SIZE=2
export VRX_REASONING_STEPS=2
export VRX_MAX_REASONING_STEPS=2
export VRX_SDE_STEPS=2
export VRX_IMAGE_SIZE=512
```

`VRX_GROUP_SIZE` 就是每个 prompt 的 GRPO 候选数 G，也决定每 rank 的 rollout batch；可以降低，但会减少组内相对排序样本和 pairwise 比较数量。降低 T 或 SDE steps 会改变 rollout budget，必须在所有 baseline 中一致。512x512 只是显存折衷，不是方法约束。

## 已验证与未验证

已验证：

- 14 个 CPU 测试通过，覆盖 parquet 解码、metadata/uid、local group sampler、格式与 GT 数量门控、共享帧、advantage、三步 GenerationStep、fake OpenAI-compatible HTTP 服务和多卡配置解析。
- 之前的四卡 512x512 mock smoke 已完成 rollout、reward、advantage、backward、optimizer update 与 checkpoint。
- 单卡 GT 7 帧真实轨迹 rollout 已生成完整 7 帧并产生非固定 mock rewards，但 rank 32 与 rank 16 都在 backward 时接近 80 GB OOM，说明主要压力来自多步 activation，不只是 LoRA 参数。

本轮按要求没有重新运行 smoke，也没有启动 GPU 作业；只执行了 CPU 单测和静态检查。


尚未验证：真实 Qwen3-VL reward server、真实 BAGEL VR-X SFT adapter 初始化、完整训练稳定性、官方七源精确采样配比，以及 paper/README 中未发布的 CLIP-variance critical-step selector。

## 关键源码位置

UniVR source of truth：

- `UniVR_RL/examples/config_emu3.yaml`
- `UniVR_RL/examples/emu3_grpo_lora.sh`
- `UniVR_RL/examples/reward_function/emu3.py`
- `UniVR_RL/verl/trainer/core_algos.py`
- `UniVR_RL/verl/trainer/ray_trainer.py`
- `UniVR_RL/verl/workers/actor/config.py`
- `UniVR_RL/verl/workers/actor/dp_actor.py`
- `UniVR_RL/verl/utils/emu3_dataset.py`

AlphaGRPO adaptation：

- `config/vrx.py`
- `alpha_grpo/dataset/vrx.py`
- `alpha_grpo/dataset/vrx_sampler.py`
- `alpha_grpo/tasks/vrx.py`
- `alpha_grpo/rewards/univr/`
- `alpha_grpo/algorithms/grpo.py`
- `alpha_grpo/stat_tracking.py`
- `tests/test_vrx.py`
