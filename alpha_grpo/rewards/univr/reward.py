# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Top-level group-aware UniVR reward composition."""

from __future__ import annotations

import os

from .format_reward import bagel_format_reward
from .global_reward import compute_global_scores
from .step_focal_reward import compute_step_focal_scores
from .vlm_client import MockVLMClient, OpenAICompatibleVLMClient


class UniVRReward:
    def __init__(
        self,
        mode="released_code",
        pairwise=True,
        mock=False,
        api_url=None,
        model=None,
        api_key=None,
        timeout=200.0,
        pairwise_timeout=2000.0,
        max_vlm_workers=128,
        format_weight=0.2,
        global_weight=2.1,
        local_weight=1.0,
        image_count_weight=0.2,
        reason_lambda=2.0,
        sampled_frames_ratio=0.5,
        seed=42,
        enable_local=True,
        expected_resolution=None,
        client=None,
    ):
        if mode == "paper":
            raise NotImplementedError(
                "UniVR's released repository does not contain the README-described "
                "CLIP-variance critical-step selector; use mode='released_code'."
            )
        if mode not in {"released_code", "vanilla"}:
            raise ValueError(f"Unknown UniVR reward mode: {mode}")
        self.mode = mode
        self.pairwise = bool(pairwise) if mode == "released_code" else False
        self.format_weight = float(format_weight)
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight) if enable_local and mode == "released_code" else 0.0
        self.image_count_weight = float(image_count_weight)
        self.reason_lambda = float(reason_lambda)
        self.sampled_frames_ratio = float(sampled_frames_ratio)
        self.seed = int(seed)
        self.expected_resolution = tuple(expected_resolution) if expected_resolution else None
        self.max_vlm_workers = max(1, int(max_vlm_workers))
        if client is not None:
            self.client = client
        elif mock or os.environ.get("UNIVR_REWARD_MOCK", "").lower() in {"1", "true", "yes"}:
            self.client = MockVLMClient()
        else:
            self.client = OpenAICompatibleVLMClient(
                api_url, model, api_key, timeout, pairwise_timeout
            )

    def __call__(self, input_data):
        records = self._records(input_data)
        format_scores = []
        image_count_scores = []
        for record in records:
            expected = record.get("expected_num_frames")
            trajectory = record.get("trajectory", [])
            gt_trajectory = record.get("gt_trajectory", [])
            score, _ = bagel_format_reward(
                trajectory,
                expected_num_frames=expected,
                expected_resolution=self.expected_resolution,
            )
            format_scores.append(score)
            image_count_scores.append(
                float(bool(gt_trajectory) and len(trajectory) == len(gt_trajectory))
            )

        valid = [format_score > 0 for format_score in format_scores]
        global_scores = compute_global_scores(
            records, self.client, pairwise=self.pairwise, valid_mask=valid,
            max_workers=self.max_vlm_workers,
        )
        if self.local_weight > 0:
            local_scores, selected_indices = compute_step_focal_scores(
                records,
                self.client,
                ratio=self.sampled_frames_ratio,
                seed=self.seed,
                valid_mask=valid,
                max_workers=self.max_vlm_workers,
            )
        else:
            local_scores = list(global_scores)
            selected_indices = [[] for _ in records]

        reason_scores = []
        overall_scores = []
        total_weight = (
            self.format_weight + self.global_weight + self.local_weight + self.image_count_weight
        )
        for index, (fmt, global_score, local_score, image_count) in enumerate(zip(
            format_scores, global_scores, local_scores, image_count_scores
        )):
            reason = (
                global_score - self.reason_lambda * abs(global_score - local_score)
                if self.local_weight > 0
                else global_score
            )
            if fmt and not image_count:
                # Released code zeroes every semantic component on a GT count mismatch.
                global_scores[index] = 0.0
                local_scores[index] = 0.0
                reason = 0.0
            reason_scores.append(reason)
            if not fmt or not image_count:
                overall = 0.0
            elif total_weight > 0:
                overall = (
                    self.format_weight * fmt
                    + (self.global_weight + self.local_weight) * reason
                    + self.image_count_weight * image_count
                ) / total_weight
            else:
                overall = reason
            overall_scores.append(overall)

        components = {
            "format": format_scores,
            "global": global_scores,
            "local_or_step_focal": local_scores,
            "reason": reason_scores,
            "image_count": image_count_scores,
            "overall": overall_scores,
        }
        metadata = {"selected_frame_indices": selected_indices}
        return overall_scores, components, metadata

    @staticmethod
    def _records(input_data):
        required = ("trajectories", "prompts", "metadata", "input_images", "gt_trajectories")
        missing = [key for key in required if key not in input_data]
        if missing:
            raise KeyError(f"Missing UniVR reward inputs: {missing}")
        count = len(input_data["trajectories"])
        uids = input_data.get("group_uids") or [
            metadata.get("uid") for metadata in input_data["metadata"]
        ]
        expected = input_data.get("expected_num_frames") or [None] * count
        return [
            {
                "uid": uids[index],
                "instruction": input_data["prompts"][index],
                "metadata": input_data["metadata"][index],
                "input_images": input_data["input_images"][index],
                "gt_trajectory": input_data["gt_trajectories"][index],
                "trajectory": input_data["trajectories"][index],
                "expected_num_frames": expected[index],
            }
            for index in range(count)
        ]


class UniVRValidationReward:
    """Released emu3_val semantics adapted to BAGEL image trajectories."""

    def __init__(
        self,
        mock=False,
        api_url=None,
        model=None,
        api_key=None,
        timeout=60.0,
        max_vlm_workers=32,
        expected_resolution=None,
        client=None,
    ):
        self.expected_resolution = (
            tuple(expected_resolution) if expected_resolution else None
        )
        self.max_vlm_workers = max(1, int(max_vlm_workers))
        if client is not None:
            self.client = client
        elif mock or os.environ.get("UNIVR_REWARD_MOCK", "").lower() in {
            "1", "true", "yes"
        }:
            self.client = MockVLMClient()
        else:
            self.client = OpenAICompatibleVLMClient(
                api_url, model, api_key, timeout=timeout, validation_prompt=True
            )

    def __call__(self, input_data):
        records = UniVRReward._records(input_data)
        format_scores = []
        for record in records:
            score, _ = bagel_format_reward(
                record.get("trajectory", []),
                expected_num_frames=record.get("expected_num_frames"),
                expected_resolution=self.expected_resolution,
            )
            format_scores.append(score)

        valid = [score > 0 for score in format_scores]
        global_scores = compute_global_scores(
            records,
            self.client,
            pairwise=False,
            valid_mask=valid,
            max_workers=self.max_vlm_workers,
        )
        overall_scores = [
            global_score if format_score > 0 else 0.0
            for format_score, global_score in zip(format_scores, global_scores)
        ]
        components = {
            "format": format_scores,
            "global": global_scores,
            "overall": overall_scores,
        }
        return overall_scores, components, {}


def univr_vrgrpo_validation(device="cpu", **kwargs):
    """Factory for UniVR's released single-sample validation reward."""
    del device
    scorer = UniVRValidationReward(**kwargs)

    def _score(**input_data):
        scores, components, metadata = scorer(input_data)
        components = dict(components)
        components["_metadata"] = metadata
        return scores, components

    return _score


def univr_vrgrpo(device="cpu", **kwargs):
    """Factory matching AlphaGRPO's reward registry convention."""
    del device
    scorer = UniVRReward(**kwargs)

    def _score(**input_data):
        scores, components, metadata = scorer(input_data)
        components = dict(components)
        components["_metadata"] = metadata
        return scores, components

    return _score

