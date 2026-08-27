# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

class PerPromptStatTracker:
    def __init__(self, global_std=False, ddof=0):
        self.global_std = global_std
        self.ddof = int(ddof)
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo', group_mean=None,):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt

            if group_mean is not None:
                mean = group_mean[prompt]
            else:
                mean = np.mean(self.stats[prompt], axis=0, keepdims=True)

            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True, ddof=self.ddof if len(rewards) > self.ddof else 0) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True, ddof=self.ddof if len(self.stats[prompt]) > self.ddof else 0) + 1e-4
            if type=='grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type=='rwr':
                # advantages[prompts == prompt] = (prompt_rewards - mean) / std
                advantages[prompts == prompt] = prompt_rewards
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current prompt
                prompt_advantages = torch.tensor(prompt_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
                # print("reward difference one group", prompt_advantages[max_idx]-prompt_advantages[min_idx])
            elif type == 'remax':
                assert group_mean is not None
                advantages[prompts == prompt] = (prompt_rewards - mean)
            elif type == 'grpo_anchor':
                baseline = np.maximum(group_mean[prompt], 
                              np.mean(self.stats[prompt], axis=0, keepdims=True))

                diff = prompt_rewards - baseline
                scale = np.sqrt(np.mean(diff**2) + 1e-8)
                advantages[prompts == prompt] = diff / scale

        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}


def main():
    tracker = PerPromptStatTracker()
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)

if __name__ == "__main__":
    main()