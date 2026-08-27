# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Global full-trajectory reward, including released Pref-GRPO mode."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def compute_global_scores(
    records, client, pairwise=True, valid_mask=None, max_workers=128
):
    valid_mask = valid_mask or [True] * len(records)
    scores = [0.0] * len(records)

    if not pairwise:
        def score_one(index):
            record = records[index]
            return index, client.score_trajectory(
                record["instruction"],
                record.get("input_images", []),
                record.get("trajectory", []),
                record.get("gt_trajectory", []),
                record.get("metadata"),
            )

        valid = [index for index in range(len(records)) if valid_mask[index]]
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
            futures = [executor.submit(score_one, index) for index in valid]
            for future in as_completed(futures):
                try:
                    index, score = future.result()
                    scores[index] = score
                except Exception:
                    pass
        return scores

    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["uid"]].append(index)

    pair_tasks = []
    wins = {}
    comparisons = {}
    for indices in groups.values():
        valid = [index for index in indices if valid_mask[index]]
        if len(valid) == 1:
            scores[valid[0]] = 0.5
            continue
        for index in valid:
            wins[index] = 0.0
            comparisons[index] = 0
        for position, left in enumerate(valid):
            for right in valid[position + 1:]:
                pair_tasks.append((left, right))

    def compare_pair(left, right):
        record = records[left]
        preference = client.compare_trajectories(
            record["instruction"],
            record.get("input_images", []),
            records[left].get("trajectory", []),
            records[right].get("trajectory", []),
            record.get("gt_trajectory", []),
            record.get("metadata"),
        )
        return left, right, preference

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(compare_pair, *task) for task in pair_tasks]
        for future in as_completed(futures):
            try:
                left, right, preference = future.result()
            except Exception:
                continue
            comparisons[left] += 1
            comparisons[right] += 1
            if preference == "A":
                wins[left] += 1.0
            elif preference == "B":
                wins[right] += 1.0
            else:
                wins[left] += 0.5
                wins[right] += 0.5

    for index in wins:
        scores[index] = (
            wins[index] / comparisons[index] if comparisons[index] else 0.5
        )
    return scores
