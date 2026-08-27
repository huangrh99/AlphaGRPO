# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

import io
import importlib.machinery
import json
import os
import sys
import tempfile
import time
import threading
import types
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "alpha_grpo"))
wandb_stub = types.ModuleType("wandb")
wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
wandb_stub.log = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", wandb_stub)

from dataset.vrx import VRXDataset
from dataset.vrx_sampler import DistributedLocalKRepeatBatchSampler
from rewards.univr.format_reward import bagel_format_reward
from rewards.univr.reward import UniVRReward, UniVRValidationReward
from rewards.univr.vlm_client import (
    OpenAICompatibleVLMClient, get_vlm_evaluation_prompt, get_vlm_validation_prompt,
)
from rewards.univr.global_reward import compute_global_scores
from stat_tracking import PerPromptStatTracker
from common import GenerationStep
from config.vrx import vrx_univr_smoke


def image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


class VRXDatasetTest(unittest.TestCase):
    def test_parquet_decode_metadata_and_stable_uid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.parquet"
            table = pa.table({
                "vlm_task_instruction": ["move the block in 2 steps"],
                "problem_image_bytes": [[image_bytes("red")]],
                "answer_image_bytes": [[image_bytes("green"), image_bytes("blue")]],
                "height": [16],
                "width": [16],
                "sub_tasks": [[{"start": 0, "end": 1, "description": "move"}]],
                "frame_subtask_mapping": [[{"frame": 0, "subtask": 0}]],
                "vlm_frame_captions": [["middle", "done"]],
                "global_summary": ["move then place"],
                "dataset_source": ["official-unit"],
            })
            pq.write_table(table, path)
            first = VRXDataset(sources={"unit": directory}, seed=7)
            second = VRXDataset(sources=ConfigDict({"unit": directory}), seed=999)
            item = first[0]
            self.assertIsInstance(item["prompt"], str)
            self.assertEqual(item["input_images"][0].mode, "RGB")
            self.assertEqual(len(item["gt_trajectory"]), 2)
            self.assertEqual(item["metadata"]["dataset_source"], "official-unit")
            self.assertEqual(item["metadata"]["dataset_partition"], "unit")
            self.assertEqual(item["metadata"]["global_summary"], "move then place")
            self.assertEqual(item["metadata"]["uid"], second[0]["metadata"]["uid"])

            resized = VRXDataset(sources={"unit": directory}, image_size=32)
            self.assertEqual(resized[0]["input_images"][0].size, (32, 32))
            self.assertEqual(resized[0]["gt_trajectory"][0].size, (32, 32))

            filtered = VRXDataset(sources={"unit": directory}, required_gt_frames=3)
            self.assertEqual(len(filtered), 0)

    def test_complete_group_stays_on_rank(self):
        dataset = list(range(8))
        samplers = [
            DistributedLocalKRepeatBatchSampler(dataset, 4, 4, 2, rank, seed=3)
            for rank in range(2)
        ]
        batches = [next(iter(sampler)) for sampler in samplers]
        self.assertEqual(len(set(batches[0])), 1)
        self.assertEqual(len(set(batches[1])), 1)
        self.assertNotEqual(batches[0][0], batches[1][0])

    def test_smoke_config_tracks_world_size_and_native_resolution(self):
        with mock.patch.dict(os.environ, {
            "WORLD_SIZE": "4",
            "VRX_SMOKE_IMAGE_SIZE": "native",
        }):
            config = vrx_univr_smoke()
        self.assertEqual(config.num_gpus, 4)
        self.assertEqual(config.total_batch_size, 8)
        self.assertEqual(config.vrx.max_samples_per_source, 4)
        self.assertIsNone(config.vrx.image_size)
        self.assertEqual(config.vrx.global_weight, 2.1)
        self.assertIn("univr_vrgrpo_validation", config.val_reward_fn)
        self.assertEqual(
            config.reward_fn["univr_vrgrpo"]["kwargs"]["max_vlm_workers"], 128
        )
        self.assertEqual(
            config.val_reward_fn["univr_vrgrpo_validation"]["kwargs"]["timeout"],
            60,
        )


class RewardTest(unittest.TestCase):
    def test_bagel_format_reward(self):
        valid = [Image.new("RGB", (16, 16), "red") for _ in range(3)]
        self.assertEqual(bagel_format_reward(valid, 3)[0], 1.0)
        self.assertEqual(bagel_format_reward(valid[:2], 3)[0], 0.0)
        self.assertEqual(bagel_format_reward([], 3)[0], 0.0)

    def test_group_reward_and_shared_frame_selection(self):
        trajectories = [
            [Image.new("RGB", (16, 16), (index * 20, frame * 20, 0)) for frame in range(3)]
            for index in range(4)
        ]
        scorer = UniVRReward(mock=True, seed=5)
        inputs = {
            "trajectories": trajectories,
            "prompts": ["task"] * 4,
            "metadata": [{"uid": "same", "dataset_source": "unit"} for _ in range(4)],
            "input_images": [[Image.new("RGB", (16, 16), "white")] for _ in range(4)],
            "gt_trajectories": [[Image.new("RGB", (16, 16), "black") for _ in range(3)] for _ in range(4)],
            "group_uids": ["same"] * 4,
            "expected_num_frames": [3] * 4,
        }
        scores, components, metadata = scorer(inputs)
        self.assertEqual(len(scores), 4)
        self.assertTrue(all(indexes == metadata["selected_frame_indices"][0]
                            for indexes in metadata["selected_frame_indices"]))
        self.assertIn("reason", components)

    def test_image_count_gate_uses_gt_length_not_planned_rollout_length(self):
        generated = [Image.new("RGB", (16, 16), "red") for _ in range(2)]
        gt = [Image.new("RGB", (16, 16), "black") for _ in range(3)]
        scorer = UniVRReward(mock=True)
        scores, components, _ = scorer({
            "trajectories": [generated],
            "prompts": ["task"],
            "metadata": [{"uid": "one"}],
            "input_images": [[Image.new("RGB", (16, 16), "white")]],
            "gt_trajectories": [gt],
            "group_uids": ["one"],
            "expected_num_frames": [2],
        })
        self.assertEqual(components["format"], [1.0])
        self.assertEqual(components["image_count"], [0.0])
        self.assertEqual(components["global"], [0.0])
        self.assertEqual(components["local_or_step_focal"], [0.0])
        self.assertEqual(components["reason"], [0.0])
        self.assertEqual(scores, [0.0])

    def test_paper_mode_is_not_silently_faked(self):
        with self.assertRaises(NotImplementedError):
            UniVRReward(mode="paper", mock=True)


class AdvantageAndStepTest(unittest.TestCase):
    def test_alpha_grpo_group_advantage(self):
        rewards = np.array([1.0, 0.5, 0.0, -0.5])
        tracker = PerPromptStatTracker(global_std=False, ddof=0)
        advantages = tracker.update(["uid"] * 4, rewards)
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(advantages.std()), 1.0, places=3)

    def test_three_image_generation_steps_have_training_fields(self):
        steps = []
        for index in range(3):
            latents = [torch.zeros(2, 1, 2, 2), torch.ones(2, 1, 2, 2)]
            log_probs = [torch.zeros(2)]
            stacked = torch.stack(latents, dim=1)
            steps.append(GenerationStep(
                step_type="image",
                images=[Image.new("RGB", (8, 8), (index, 0, 0)) for _ in range(2)],
                latents=stacked[:, :-1],
                next_latents=stacked[:, 1:],
                image_log_probs=torch.stack(log_probs, dim=1),
                sde_window=(0, 1),
                task_type="edit",
            ))
        self.assertEqual(len(steps), 3)
        self.assertTrue(all(step.step_type == "image" for step in steps))
        self.assertTrue(all(step.latents is not None and step.next_latents is not None
                            and step.image_log_probs is not None and step.sde_window is not None
                            for step in steps))


class FakeVLMHandler(BaseHTTPRequestHandler):
    last_payload = None

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        type(self).last_payload = json.loads(self.rfile.read(length))
        payload = {
            "choices": [{
                "message": {
                    "content": "[[TASK_SCORE: 0.80]]\n[[VISUAL_SCORE: 0.60]]"
                }
            }]
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass


class RewardServerTest(unittest.TestCase):
    def test_openai_compatible_fake_server(self):
        server = HTTPServer(("127.0.0.1", 0), FakeVLMHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleVLMClient(
                api_url=f"http://127.0.0.1:{server.server_port}/v1",
                model="fake",
                api_key="test",
            )
            image = Image.new("RGB", (8, 8), "red")
            score = client.score_trajectory("task", [image], [image], [image])
            self.assertAlmostEqual(score, 0.7)
            payload = FakeVLMHandler.last_payload
            self.assertEqual(len(payload["messages"]), 1)
            self.assertEqual(payload["messages"][0]["role"], "user")
            content = payload["messages"][0]["content"]
            labels = [part.get("text", "") for part in content if part["type"] == "text"]
            gt_pos = next(i for i, value in enumerate(labels) if "Ground Truth" in value)
            ref_pos = next(i for i, value in enumerate(labels) if "Input Reference" in value)
            gen_pos = next(i for i, value in enumerate(labels) if "Model Generated" in value)
            self.assertLess(gt_pos, ref_pos)
            self.assertLess(ref_pos, gen_pos)
            self.assertIn(
                "You are a reward model for multimodal generative models",
                labels[0],
            )
            self.assertAlmostEqual(
                client.score_frame("task", [image], image, image, 0), 0.7
            )
            with mock.patch.object(client, "_complete", return_value="[[PREFER: B]]"):
                self.assertEqual(
                    client.compare_trajectories("task", [image], [image], [image], [image]),
                    "B",
                )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()



class RewardCallAlignmentTest(unittest.TestCase):
    def test_validation_uses_official_task_specific_rubric(self):
        train_prompt, _ = get_vlm_evaluation_prompt("pickup at supermarket", "agibot")
        val_prompt, _ = get_vlm_validation_prompt("pickup at supermarket", "agibot")
        self.assertNotIn("robot task for picking up items", train_prompt)
        self.assertIn("robot task for picking up items", val_prompt)

    def test_pairwise_uses_official_timeout_and_single_user_message(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": "[[PREFER: A]]"}}]
        }
        client = OpenAICompatibleVLMClient(
            api_url="http://reward.test/v1",
            model="fake",
            api_key="test",
            timeout=200,
            pairwise_timeout=2000,
        )
        image = Image.new("RGB", (8, 8), "red")
        with mock.patch(
            "rewards.univr.vlm_client.requests.post", return_value=response
        ) as post:
            preference = client.compare_trajectories(
                "task", [image], [image], [image], [image]
            )
        self.assertEqual(preference, "A")
        self.assertEqual(post.call_args.kwargs["timeout"], 2000)
        messages = post.call_args.kwargs["json"]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user"])

    def test_validation_has_format_gate_but_no_gt_count_gate(self):
        class AbsoluteClient:
            def score_trajectory(self, *args, **kwargs):
                return 0.8

        image = Image.new("RGB", (8, 8), "red")
        scorer = UniVRValidationReward(client=AbsoluteClient())
        scores, components, _ = scorer({
            "trajectories": [[image, image]],
            "prompts": ["task"],
            "metadata": [{"uid": "one"}],
            "input_images": [[image]],
            "gt_trajectories": [[image, image, image]],
            "group_uids": ["one"],
            "expected_num_frames": [2],
        })
        self.assertEqual(components["format"], [1.0])
        self.assertEqual(scores, [0.8])

    def test_pairwise_calls_are_concurrent(self):
        class ConcurrentClient:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.lock = threading.Lock()

            def compare_trajectories(self, *args, **kwargs):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return "TIE"

        image = Image.new("RGB", (8, 8), "red")
        records = [{
            "uid": "group",
            "instruction": "task",
            "input_images": [image],
            "trajectory": [image],
            "gt_trajectory": [image],
            "metadata": {},
        } for _ in range(4)]
        client = ConcurrentClient()
        scores = compute_global_scores(records, client, pairwise=True, max_workers=4)
        self.assertEqual(scores, [0.5] * 4)
        self.assertGreater(client.max_active, 1)

if __name__ == "__main__":
    unittest.main()

