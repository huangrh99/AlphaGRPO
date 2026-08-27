# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dataset classes and dataloader construction for AlphaGRPO training."""

import os
import json
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, BatchSampler
from .vrx import VRXDataset
from .vrx_sampler import DistributedLocalKRepeatBatchSampler


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class TextPromptWithQuestionDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        quality_questions = []
        for question in self.metadatas[idx]['quality_questions']:
            quality_questions.append(question)
        return {"prompt": self.metadatas[idx]['prompt'], "metadata": {"semantic_questions": self.metadatas[idx]["semantic_questions"], "quality_questions": quality_questions}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class ImageTextDataset(Dataset):
    def __init__(self, dataset, image_root, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.jsonl')
        self.image_root = image_root
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
            self.images = [item['image'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_root, self.images[idx]))
        return {"prompt": self.prompts[idx], "image": image, "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        images = [example["image"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas, images


class DistributedKRepeatBatchSampler(BatchSampler):
    """
    Distributed K-repeat batch sampler.
    Each unique sample is repeated k times, then distributed across GPUs.
    """

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # in one batch, how many unique prompts
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        dataset_size = len(self.dataset)
        num_rounds = dataset_size // self.m

        all_indices = torch.randperm(dataset_size, generator=g)[: dataset_size // self.m * self.m]
        all_indices = all_indices.view(num_rounds, self.m)

        for round_indices in all_indices:
            repeated_indices = round_indices.unsqueeze(-1).repeat(1, self.k).flatten()
            shuffled_indices = repeated_indices[torch.randperm(len(repeated_indices), generator=g)].tolist()

            assert torch.unique(torch.as_tensor(
                repeated_indices)).numel() == self.m, f"Unique prompts: {torch.unique(torch.as_tensor(repeated_indices)).numel()}, expected: {self.m}"

            batch = shuffled_indices[self.rank::self.num_replicas]
            yield batch

    def __len__(self):
        return len(self.dataset) // self.m

    def set_epoch(self, epoch):
        self.epoch = epoch


# ====== Mixed-Task Dataset & Sampler ======

class MixedTaskDataset(Dataset):
    """Concatenated dataset for mixed-task training.

    Wraps multiple sub-datasets into one. Each item carries a ``_task`` tag
    so that the collate_fn can propagate the task name into the batch.
    MixedTaskBatchSampler guarantees that every batch is homogeneous
    (all items come from the same sub-dataset / task).
    """

    def __init__(self, sub_datasets, task_names):
        self.sub_datasets = sub_datasets
        self.task_names = task_names
        self.offsets = []
        offset = 0
        for ds in sub_datasets:
            self.offsets.append(offset)
            offset += len(ds)
        self.total_length = offset

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Find which sub-dataset this global index belongs to
        ds_idx = len(self.offsets) - 1
        for i in range(len(self.offsets) - 1):
            if idx < self.offsets[i + 1]:
                ds_idx = i
                break
        local_idx = idx - self.offsets[ds_idx]
        item = self.sub_datasets[ds_idx][local_idx]
        item['_task'] = self.task_names[ds_idx]
        return item

    @staticmethod
    def collate_fn(examples):
        """Collate with a task-name prefix: ``(task, prompts, metadatas[, images])``."""
        task = examples[0]['_task']
        prompts = [ex["prompt"] for ex in examples]
        metadatas = [ex["metadata"] for ex in examples]
        if "image" in examples[0]:
            images = [ex["image"] for ex in examples]
            return task, prompts, metadatas, images
        return task, prompts, metadatas


class MixedTaskBatchSampler(BatchSampler):
    """Distributed K-repeat batch sampler for mixed-task training.

    Each batch is **homogeneous** — all indices come from the same sub-dataset
    (and therefore the same task).  Batches from different datasets are
    interleaved based on ``weights``.

    Internally, for each sub-dataset the sampler applies the same K-repeat
    logic as ``DistributedKRepeatBatchSampler``:
    - Shuffle the sub-dataset indices.
    - Partition into rounds of ``m`` unique prompts.
    - Each round is K-repeated, shuffled, and split across GPUs.
    - The resulting per-round batches are collected, then interleaved
      across datasets according to sampling weights.
    """

    def __init__(self, sub_datasets, offsets, batch_size, k,
                 num_replicas, rank, weights=None, seed=0):
        self.sub_datasets = sub_datasets
        self.offsets = offsets
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.weights = weights or [1.0] * len(sub_datasets)
        self.seed = seed
        self.epoch = 0

        self.total_samples = num_replicas * batch_size
        assert self.total_samples % k == 0, (
            f"k cannot divide n*b: k={k}, num_replicas={num_replicas}, batch_size={batch_size}"
        )
        self.m = self.total_samples // k  # unique prompts per global batch

    def _build_batches(self, g):
        """Pre-compute K-repeat batches per sub-dataset, then interleave."""
        per_ds_batches = []

        for ds_idx, (dataset, offset) in enumerate(zip(self.sub_datasets, self.offsets)):
            ds_size = len(dataset)
            num_rounds = ds_size // self.m
            if num_rounds == 0:
                per_ds_batches.append([])
                continue

            local_indices = torch.randperm(ds_size, generator=g)[:num_rounds * self.m]
            local_indices = local_indices.view(num_rounds, self.m)

            batches = []
            for round_indices in local_indices:
                repeated = round_indices.unsqueeze(-1).repeat(1, self.k).flatten()
                shuffled = repeated[torch.randperm(len(repeated), generator=g)]
                global_indices = (shuffled + offset).tolist()
                batch = global_indices[self.rank::self.num_replicas]
                batches.append(batch)
            per_ds_batches.append(batches)

        # Determine how many batches to draw from each dataset.
        # Weight controls relative frequency; capped by available batches.
        total_weight = sum(self.weights)
        total_available = sum(len(b) for b in per_ds_batches)

        selected = []
        for ds_idx, batches in enumerate(per_ds_batches):
            if not batches:
                continue
            ratio = self.weights[ds_idx] / total_weight
            target = max(1, round(total_available * ratio))
            target = min(target, len(batches))
            selected.extend(batches[:target])

        # Shuffle interleaved batches
        perm = torch.randperm(len(selected), generator=g)
        return [selected[i] for i in perm]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        for batch in self._build_batches(g):
            yield batch

    def __len__(self):
        total_weight = sum(self.weights)
        total_available = sum(len(ds) // self.m for ds in self.sub_datasets)
        count = 0
        for ds_idx, ds in enumerate(self.sub_datasets):
            available = len(ds) // self.m
            ratio = self.weights[ds_idx] / total_weight
            target = max(1, round(total_available * ratio))
            count += min(target, available)
        return count

    def set_epoch(self, epoch):
        self.epoch = epoch


# ====== Dataset Registry ======

DATASET_MAP = {
    "general_ocr": (TextPromptDataset, {}),
    "geneval": (GenevalPromptDataset, {}),
    "image_text": (ImageTextDataset, {"image_root": True}),
    "dvreward": (TextPromptWithQuestionDataset, {}),
    "vrx": (VRXDataset, {}),
}


def _build_single_dataset(prompt_fn, dataset_path, image_root=None, split='train', dataset_kwargs=None):
    """Instantiate one dataset from a prompt_fn key."""
    if prompt_fn not in DATASET_MAP:
        raise NotImplementedError(
            f"Unsupported prompt_fn: {prompt_fn}. Available: {list(DATASET_MAP.keys())}"
        )
    dataset_cls, required_opts = DATASET_MAP[prompt_fn]
    required_opts = dict(required_opts)
    if required_opts.get("image_root"):
        assert image_root is not None
        required_opts.update({"image_root": image_root})
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_kwargs = dict(dataset_kwargs or {})
    if prompt_fn == "vrx":
        fallback_path = dataset_path
        if fallback_path and not os.path.isabs(fallback_path):
            fallback_path = os.path.join(base_dir, fallback_path)
        return dataset_cls(fallback_path, split=split, **dataset_kwargs), dataset_cls
    return dataset_cls(
        os.path.join(base_dir, dataset_path), split=split, **required_opts, **dataset_kwargs
    ), dataset_cls


# ====== Public Interface ======

def build_dataloader(config, accelerator) -> Tuple[DataLoader, DataLoader]:
    """Build train and test dataloaders.

    If ``config.mix_dataset`` exists (list of sub-dataset specs), builds a
    mixed-task dataloader.  Otherwise falls back to single-dataset mode
    using ``config.prompt_fn``.
    """
    if getattr(config, 'mix_dataset', None):
        return _build_mixed_dataloader(config, accelerator)
    return _build_single_dataloader(config, accelerator)


def _build_single_dataloader(config, accelerator):
    prompt_fn = config.prompt_fn
    dataset_cls = DATASET_MAP[prompt_fn][0]

    train_kwargs = {}
    test_kwargs = {}
    if prompt_fn == "vrx":
        vrx = config.vrx
        train_kwargs = {
            "sources": getattr(vrx, "train_sources", None),
            "max_samples_per_source": getattr(vrx, "max_samples_per_source", None),
            "image_size": getattr(vrx, "image_size", None),
            "required_gt_frames": getattr(vrx, "required_gt_frames", None),
            "seed": config.seed,
        }
        test_kwargs = {
            "sources": getattr(vrx, "val_sources", None) or getattr(vrx, "train_sources", None),
            "max_samples_per_source": getattr(vrx, "max_val_samples_per_source", None),
            "image_size": getattr(vrx, "image_size", None),
            "required_gt_frames": getattr(vrx, "required_gt_frames", None),
            "seed": config.seed,
        }
    train_dataset, dataset_cls = _build_single_dataset(
        prompt_fn, config.dataset, getattr(config, 'image_root', None), 'train', train_kwargs)
    test_dataset, _ = _build_single_dataset(
        prompt_fn, config.dataset, getattr(config, 'image_root', None), 'test', test_kwargs)

    sampler_cls = (
        DistributedLocalKRepeatBatchSampler
        if prompt_fn == "vrx"
        else DistributedKRepeatBatchSampler
    )
    train_sampler = sampler_cls(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed
    )

    num_workers_train = config.train.dataloader_num_workers
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers_train,
        collate_fn=dataset_cls.collate_fn,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=dataset_cls.collate_fn,
        shuffle=False,
        num_workers=config.train.dataloader_num_workers,
    )

    return train_dataloader, test_dataloader


def _build_mixed_dataloader(config, accelerator):
    """Build mixed-task train dataloader + a single test dataloader.

    Config format::

        config.mix_dataset = [
            {
                "task": "t2i",
                "prompt_fn": "hard_prompt_v2",
                "dataset": "/path/to/data",
                "weight": 0.6,
            },
            {
                "task": "ti2i",
                "prompt_fn": "hard_prompt_image_v1",
                "dataset": "/path/to/data",
                "image_root": "/path/to/images",
                "weight": 0.4,
            },
        ]

    Each item reuses the same ``prompt_fn`` / ``dataset`` / ``image_root``
    fields as the single-dataset config.  ``task`` specifies which task to
    dispatch to, and ``weight`` controls relative sampling frequency.
    """
    sub_datasets_train = []
    task_names = []
    weights = []

    for item in config.mix_dataset:
        task_name = item['task']
        prompt_fn = item['prompt_fn']
        dataset_path = item['dataset']
        image_root = item.get('image_root', None)
        weight = item.get('weight', 1.0)

        train_ds, _ = _build_single_dataset(prompt_fn, dataset_path, image_root, 'train')

        sub_datasets_train.append(train_ds)
        task_names.append(task_name)
        weights.append(weight)

    mixed_train = MixedTaskDataset(sub_datasets_train, task_names)

    train_sampler = MixedTaskBatchSampler(
        sub_datasets=sub_datasets_train,
        offsets=mixed_train.offsets,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        weights=weights,
        seed=42,
    )

    train_dataloader = DataLoader(
        mixed_train,
        batch_sampler=train_sampler,
        num_workers=4,
        collate_fn=MixedTaskDataset.collate_fn,
        shuffle=False,
    )

    # Test dataloader: use config.train.eval_task to pick which sub-dataset,
    # falling back to the first sub-dataset.
    eval_task = getattr(config.train, 'eval_task', None)
    eval_item = config.mix_dataset[0]  # default
    if eval_task:
        for item in config.mix_dataset:
            if item['task'] == eval_task:
                eval_item = item
                break

    test_dataset, test_cls = _build_single_dataset(
        eval_item['prompt_fn'], eval_item['dataset'],
        eval_item.get('image_root'), 'test'
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=test_cls.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    return train_dataloader, test_dataloader