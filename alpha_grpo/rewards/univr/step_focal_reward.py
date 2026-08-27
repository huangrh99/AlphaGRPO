# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Released-code sampled-frame local reward with group-shared indices."""

import hashlib
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def shared_frame_indices(uid, frame_count, ratio=0.5, seed=42):
    if frame_count <= 0:
        return []
    digest = hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    count = max(1, int(frame_count * ratio))
    return sorted(rng.sample(range(frame_count), min(count, frame_count)))


def compute_step_focal_scores(
    records, client, ratio=0.5, seed=42, valid_mask=None, max_workers=128
):
    valid_mask = valid_mask or [True] * len(records)
    scores = [0.0] * len(records)
    selected = [[] for _ in records]
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["uid"]].append(index)

    frame_tasks = []
    for uid, indices in groups.items():
        gt = records[indices[0]].get("gt_trajectory", [])
        group_indices = shared_frame_indices(uid, len(gt), ratio=ratio, seed=seed)
        for index in indices:
            selected[index] = group_indices
            trajectory = records[index].get("trajectory", [])
            if not valid_mask[index] or len(trajectory) != len(gt) or not group_indices:
                continue
            for frame_index in group_indices:
                frame_tasks.append((index, frame_index))

    def score_frame(index, frame_index):
        record = records[index]
        return index, client.score_frame(
            record["instruction"],
            record.get("input_images", []),
            record["trajectory"][frame_index],
            record["gt_trajectory"][frame_index],
            frame_index,
            record.get("metadata"),
        )

    per_record = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(score_frame, *task) for task in frame_tasks]
        for future in as_completed(futures):
            try:
                index, score = future.result()
                per_record[index].append(score)
            except Exception:
                pass

    for index, frame_scores in per_record.items():
        if frame_scores:
            scores[index] = sum(frame_scores) / len(frame_scores)
    return scores, selected
