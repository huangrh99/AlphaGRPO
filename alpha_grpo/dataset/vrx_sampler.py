# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed sampler that keeps each complete GRPO group on one rank."""

import torch
from torch.utils.data import BatchSampler


class DistributedLocalKRepeatBatchSampler(BatchSampler):
    """Assign unique prompts to ranks, then repeat each prompt locally K times."""

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0, drop_last=True):
        if batch_size % k:
            raise ValueError(
                f"VR-X local group sampling requires per-rank batch_size divisible by k: "
                f"batch_size={batch_size}, k={k}"
            )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.groups_per_rank = self.batch_size // self.k
        self.unique_per_round = self.groups_per_rank * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset), generator=generator)
        usable = len(order) // self.unique_per_round * self.unique_per_round
        rounds = order[:usable].view(-1, self.unique_per_round)
        start = self.rank * self.groups_per_rank
        for round_indices in rounds:
            local = round_indices[start:start + self.groups_per_rank]
            repeated = local.unsqueeze(-1).repeat(1, self.k).flatten()
            yield repeated[torch.randperm(len(repeated), generator=generator)].tolist()

    def __len__(self):
        return len(self.dataset) // self.unique_per_round

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

