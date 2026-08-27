# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict
import math
from functools import partial

import time
import os
import threading
import socket
import http.server
import socketserver
import uuid
import subprocess
from functools import partial
import re


def _make_openai_client():
    """Create OpenAI/AzureOpenAI client from env vars with shared config."""
    import openai
    import httpx

    base_url = os.environ.get("OPENAI_API_URL", None)
    ak = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY") or "placeholder"
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-2025-04-14")

    http_client = httpx.Client(
        limits=httpx.Limits(max_connections=2000, max_keepalive_connections=500),
        timeout=httpx.Timeout(6000.0, connect=600.0, read=600.0, write=600.0, pool=600.0),
        proxies=None,
    )

    if 'gpt' in model_name:
        client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            api_key=ak,
            http_client=http_client,
        )
    else:
        client = openai.OpenAI(
            base_url=base_url,
            api_key=ak,
            http_client=http_client,
        )

    return client, model_name


def _tensor_to_pil_images(images):
    """Convert NCHW float tensor to list of PIL Images."""
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    images = images.permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(img) for img in images]


def _tensor_to_numpy_nhwc(images):
    """Convert NCHW float tensor to NHWC uint8 numpy array."""
    images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    return images.permute(0, 2, 3, 1).numpy()


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        else:
            images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from rewards.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score():
    from rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def siglip_score(device='cuda'):
    from rewards.siglip_scorer import SiglipScorer

    scorer = SiglipScorer(device=device).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from rewards.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from rewards.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from rewards.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def ocr_score(device):
    from rewards.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_numpy_nhwc(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def video_ocr_score(device):
    from rewards.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = _tensor_to_numpy_nhwc(images)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = _tensor_to_numpy_nhwc(images)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_numpy_nhwc(images)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_image_text}"

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    url = os.environ.get("UNIFIED_REWARD_URL", "http://127.0.0.1:17140/v1")
    model_name = os.environ.get("UNIFIED_REWARD_MODEL", "UnifiedReward-7b-v1.5")
    max_workers = int(os.environ.get("UNIFIED_REWARD_MAX_CONCURRENT", "8"))
    client = OpenAI(base_url=url, api_key="flowgrpo")

    def evaluate_image(prompt, image):
        question = (
            f"<image>\nYou are given a text caption and a generated image based on that caption. "
            f"Your task is to evaluate this image based on two key criteria:\n"
            f"1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
            f"Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
            f"2. Overall Image Quality: Examine the visual quality of this image, including clarity, "
            f"detail preservation, color accuracy, and overall aesthetic appeal.\n"
            f"Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
            f"Your task is provided as follows:\nText Caption: [{prompt}]"
        )
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                            {"type": "text", "text": question},
                        ],
                    }],
                    temperature=0,
                    max_tokens=512,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[unifiedreward] attempt {attempt+1} failed: {e}")
                time.sleep(1)
        return ""

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        elif isinstance(images, np.ndarray):
            images = [Image.fromarray(img) for img in images]
        else:
            images = [img if isinstance(img, Image.Image) else Image.fromarray(img) for img in images]
        images = [img.resize((512, 512)) for img in images]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            text_outputs = list(executor.map(evaluate_image, prompts, images))

        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn


def hpsv3_score_remote(device):
    """Submits images to HPSv3 server and computes a reward.

    Server protocol (pickle over HTTP, same as deqa/geneval):
      Request:  {"images": [jpeg_bytes], "prompts": [str]}
      Response: {"outputs": [float]}

    Launch server:  bash scripts/serve_hpsv3.sh [host] [port] [device]
    Default endpoint: http://127.0.0.1:18087
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 32  # HPSv3 (Qwen2-VL 7B) is heavier than CLIP-based scorers
    url = os.environ.get("HPSV3_SERVER_URL", "http://127.0.0.1:18087")
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_numpy_nhwc(images)
        images_batched = np.array_split(images, max(1, int(np.ceil(len(images) / batch_size))))
        prompts_batched = np.array_split(prompts, max(1, int(np.ceil(len(prompts) / batch_size))))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            data = {
                "images": jpeg_images,
                "prompts": list(prompt_batch),
            }
            data_bytes = pickle.dumps(data)

            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)
            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn


def hpsv2_score():
    import hpsv2

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        # scores = [hpsv2.score(img, prompt, hps_version="v2.1")[0] for img, prompt in zip(images, prompts)]
        return torch.as_tensor(scores), {}
    return _fn


def _spectrareward_to_pil_list(images):
    """Normalize tensor/ndarray/PIL inputs to a list of RGB PIL images."""
    if isinstance(images, torch.Tensor):
        return _tensor_to_pil_images(images)
    if isinstance(images, np.ndarray):
        if images.dtype != np.uint8:
            images = (np.clip(images, 0, 1) * 255).round().astype(np.uint8)
        return [Image.fromarray(image).convert("RGB") for image in images]
    return [
        image.convert("RGB") if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
        for image in images
    ]


def _spectrareward_remote_fn(url, model_id):
    """Pickle-over-HTTP client for a remote SpectraReward server.

    Protocol (matches scripts/mllm_server.py):
      Request:  pickle.dumps({"images": [jpeg_bytes], "prompts": [str]})
      Response: pickle.dumps({"outputs": [float]})

    The server applies prompt prefix/suffix/user-instruction/EOS handling, so the
    client only ships images and prompts.
    """
    import pickle
    from io import BytesIO

    import requests
    from requests.adapters import HTTPAdapter, Retry

    batch_size = 32
    sess = requests.Session()
    # Reward servers are internal endpoints; bypass any corp proxy from the env.
    sess.trust_env = False
    sess.proxies = {"http": None, "https": None}
    retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        pil_images = _spectrareward_to_pil_list(images)
        prompts = list(prompts)
        all_scores = []
        for start in range(0, len(pil_images), batch_size):
            img_batch = pil_images[start:start + batch_size]
            prompt_batch = prompts[start:start + batch_size]
            jpeg_images = []
            for img in img_batch:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                jpeg_images.append(buf.getvalue())
            response = sess.post(
                url,
                data=pickle.dumps({"images": jpeg_images, "prompts": prompt_batch}),
                timeout=600,
            )
            all_scores += pickle.loads(response.content)["outputs"]
        return all_scores, {"spectrareward_model": [model_id] * len(prompts)}

    return _fn


def spectrareward_score(device):
    """External SpectraReward with a frozen MLLM reward model.

    Two modes, selected by env:
      - Remote: if SPECTRAREWARD_URL (or SPECTRAREWARD_URL_<SLUG>) is set, ship
        images/prompts to a SpectraReward server (scripts/mllm_server.py).
        Recommended for large reward MLLMs that should not share a GPU with BAGEL.
      - In-process: otherwise load the reward MLLM locally with lazy GPU offload.

    Configure via environment variables:
      SPECTRAREWARD_MODEL_ID, SPECTRAREWARD_URL[_<SLUG>],
      SPECTRAREWARD_PROMPT_PREFIX, SPECTRAREWARD_PROMPT_SUFFIX,
      SPECTRAREWARD_USER_INSTRUCTION, SPECTRAREWARD_EXCLUDE_EOS,
      SPECTRAREWARD_LAZY_GPU, SPECTRAREWARD_ATTN_IMPLEMENTATION.
    """
    model_id = os.environ.get("SPECTRAREWARD_MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct")

    slug = model_id.replace("/", "_").replace("-", "_").replace(".", "_").upper()
    url = os.environ.get(f"SPECTRAREWARD_URL_{slug}") or os.environ.get("SPECTRAREWARD_URL")
    if url:
        return _spectrareward_remote_fn(url, model_id)

    from rewards.spectrareward import SpectraRewardScorer

    lazy_gpu_default = torch.device(device).type == "cuda"
    scorer = SpectraRewardScorer(
        model_id=model_id,
        device=str(device),
        prompt_prefix=os.environ.get("SPECTRAREWARD_PROMPT_PREFIX", ""),
        prompt_suffix=os.environ.get("SPECTRAREWARD_PROMPT_SUFFIX", ""),
        user_instruction=os.environ.get("SPECTRAREWARD_USER_INSTRUCTION", ""),
        exclude_eos=_env_flag("SPECTRAREWARD_EXCLUDE_EOS", True),
        lazy_gpu=_env_flag("SPECTRAREWARD_LAZY_GPU", lazy_gpu_default),
        attn_implementation=os.environ.get("SPECTRAREWARD_ATTN_IMPLEMENTATION", "sdpa"),
    )

    def _fn(images, prompts, metadata):
        scores = scorer(_spectrareward_to_pil_list(images), prompts)
        return scores, {"spectrareward_model": [model_id] * len(prompts)}

    return _fn


def viescorer_t2i(device, backbone="qwen3vl", task='t2i'):
    """Submits images to VIEScorer and computes a reward.
    We use Qwen3VL-30B-A3B-Thinking as the reward model.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from viescore import VIEScore

    os.environ["HTTP_PROXY"] =  ""
    os.environ["http_proxy"] =  ""
    os.environ["https_proxy"] = ""
    os.environ['no_proxy'] = ''

    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys', 'secret.env')
    vie_score = VIEScore(backbone=backbone, task=task, key_path=key_path)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)

        all_scores = []
        all_metas = []

        from concurrent.futures import ThreadPoolExecutor
        evaluate_func_with_arg = partial(vie_score.evaluate, extract_all_score=False)
        with ThreadPoolExecutor(max_workers=16) as executor:
            results_dicts = executor.map(evaluate_func_with_arg, images, prompts)

            for results_dict in results_dicts:
                semantics_score = min(results_dict['SC']['score'])
                quality_score = min(results_dict['PQ']['score'])
                overall_score = math.sqrt(semantics_score * quality_score)
                semantics_score_reasoning = results_dict['SC']['reasoning']
                quality_score_reasoning = results_dict['PQ']['reasoning']

                meta = {
                    "semantics_score": semantics_score,
                    "semantics_score_reasoning": semantics_score_reasoning,
                    "quality_score": quality_score,
                    "quality_score_reasoning": quality_score_reasoning,
                    "overall_score": overall_score,
                }
            
                all_scores.append(overall_score / 10.0)
                all_metas.append(meta)

        return all_scores, all_metas

    return _fn


def viescorer_t2i_compare(device, backbone="qwen3vl", task='t2i_compare'):
    return viescorer_t2i(device, backbone=backbone, task=task)


def decompositional_verifiable_reward(device, backbone="qwen3vl", task='t2i', use_logprobs=True, 
                   temperature=0.0, score_method='geometric'):
    """Submits images to VIEScorer and computes a reward.
    We use Qwen3VL-30B-A3B-Thinking as the reward model.
    """
    from io import BytesIO
    import base64
    import json
    from concurrent.futures import ThreadPoolExecutor
    import requests

    os.environ["HTTP_PROXY"] =  ""
    os.environ["http_proxy"] =  ""
    os.environ["https_proxy"] = ""
    os.environ['no_proxy'] = ''

    system_prompt = '''You are an expert visual auditor. Your task is to strictly evaluate an AI-generated image against a specific question. Answer only with 'yes' or 'no'. Do not give other outputs or punctuation marks. If the subjects in the question don't exist, answer 'no'.'''

    client, model_name = _make_openai_client()

    SERVER_IP = os.environ.get("FILE_SERVER_ADDR", "127.0.0.1")
    SIDECAR_PORT =  os.environ.get("FILE_SERVER_PORT", "8000")

    def check_server_availability(timeout=1.0):
        try:
            # Method A: Socket probe (fastest, only checks if the port is open)
            # Note: for socket, do NOT wrap IPv6 addresses with [], pass the raw string directly
            sock = socket.socket(socket.AF_INET6 if ":" in SERVER_IP else socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((SERVER_IP, int(SIDECAR_PORT)))
            sock.close()
            return result == 0
            
        except Exception as e:
            print(f"[Warning] File server check failed: {e}")
            return False

    USE_REMOTE_SERVER = check_server_availability()

    def upload_image_to_server(image):
        for _ in range(5):
            try:
                filename = f"{uuid.uuid4().hex}.jpg"
                
                # 1. Convert image to binary stream
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='JPEG', quality=95)
                img_bytes = img_byte_arr.getvalue()
                
                # 2. Build upload URL (handle IPv6 if necessary)
                host = f"[{SERVER_IP}]" if ":" in SERVER_IP else SERVER_IP
                upload_url = f"http://{host}:{SIDECAR_PORT}/upload?filename={filename}"
                
                # 3. Send POST request (raw binary body)
                resp = requests.post(upload_url, data=img_bytes, proxies={"http": None, "https": None})
                resp.raise_for_status()
                
                # Server returns: file:///dev/shm/eval_images/xxx.jpg
                remote_file_path = resp.text.strip()
                return filename, remote_file_path
            except Exception as e:
                print(f"upload_image_to_server failed: {e}")
                time.sleep(1)
        return None, None
    
    def delete_remote_image(filename):
        try:
            host = f"[{SERVER_IP}]" if ":" in SERVER_IP else SERVER_IP
            delete_url = f"http://{host}:{SIDECAR_PORT}/delete?filename={filename}"
            requests.delete(delete_url, proxies={"http": None, "https": None})
        except:
            pass

    def encode_pil_image(pil_image):
        # Create an in-memory binary stream
        image_stream = BytesIO()
        
        # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
        pil_image.save(image_stream, format='JPEG')
        
        # Get the binary data from the stream and encode it as base64
        image_data = image_stream.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        return base64_image
        
    def prepare_prompt(text_prompt: str = "", image_links = []):
        prompt_content = []

        if image_links is not None:
            if  not isinstance(image_links, list):
                image_links = [image_links]

            for image_link in image_links:
                if isinstance(image_link, str) and image_link.startswith("http://"):
                    encoded_img_url = image_link
                elif isinstance(image_link, str) and image_link.startswith("file://"):
                    encoded_img_url = image_link.replace('file://', '')
                else:   
                    if isinstance(image_link, str):
                        image = Image.open(image_link)
                    else:
                        image = image_link
                    encoded_img_url = f"data:image/jpeg;base64,{encode_pil_image(image)}"
                visual_dict = {
                        "type": "image_url",
                        "image_url": {"url": encoded_img_url}
                }
                prompt_content.append(visual_dict)

        text_dict = {
                    "type": "text",
                    "text": text_prompt
                }
        prompt_content.append(text_dict)
        return prompt_content

    def calculate_confidence_score(completion):
        """
        Calculate a confidence score for a Yes/No answer based on top logprobs.

        This function extracts the top logprob tokens from the completion object,
        looks for tokens corresponding to "Yes" and "No", converts their logprobs
        into probabilities, and returns a normalized score:

            score = P(Yes) / (P(Yes) + P(No))

        If neither token appears in the top-k candidates, it falls back to checking
        the generated text directly.

        Args:
            completion: The raw response object returned by the model API.

        Returns:
            float: A confidence score between 0.0 and 1.0.
                Closer to 1 → confident "Yes",
                closer to 0 → confident "No".
        """

        # Extract the top logprobs for the first generated token
        top_logprobs = completion['choices'][0]['logprobs']['content'][0]['top_logprobs']
        generated_text = (
            completion["choices"][0]["message"]["content"].lower()
                .strip()
                .replace(".", "")
                .replace(",", "")
                .replace("?", "")
                .replace("!", "")
            )

        probs_yes = []
        probs_no = []
        
        for item in top_logprobs:
            token = item['token']
            logprob = item['logprob']
            
            # 清洗 Token
            clean_token = token.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
            
            if clean_token == 'yes':
                probs_yes.append(math.exp(logprob))
            elif clean_token == 'no':
                probs_no.append(math.exp(logprob))
        
        sum_prob_yes = sum(probs_yes)
        sum_prob_no = sum(probs_no)

        if sum_prob_yes + sum_prob_no > 0.:
            final_score = sum_prob_yes / (sum_prob_yes + sum_prob_no)
            return final_score
        else:
            return 1.0 if generated_text == "yes" else 0.0

    def chat(prompt, images = None, system_prompt=None, max_tokens = 4094, 
             temperature=0.3, top_p=1.0, return_logprobs=False, **kwargs):
        for _ in range(10):
            try:
                messages = []
                if system_prompt is not None:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({
                    "role": "user",
                    "content": prepare_prompt(text_prompt=prompt, image_links=images)
                })

                if 'extra_headers' in kwargs:
                    kwargs['extra_headers'].update({"X-TT-LOGID": ""})
                else:
                    kwargs['extra_headers'] = {"X-TT-LOGID": ""}

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    logprobs=True,
                    top_logprobs=5,
                    **kwargs,
                    timeout=180000000,
                )

                completion = completion.json()
                if isinstance(completion, str):
                    completion = json.loads(completion)
                if return_logprobs:
                    score = calculate_confidence_score(completion)
                else:
                    answer = completion['choices'][0]['message']['content']
                    answer = answer.lower().strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
                    score = answer == 'yes'

                return score
            except Exception as e:
                print(e)
                continue
        print('Warning: Failed to get response after 10 times, return 0.')
        return 0.
     

    def evalue_one_image(image, prompt, metadata):
        if USE_REMOTE_SERVER:
            filename, img_url = upload_image_to_server(image)
            if filename is None:
                filename = img_url = image
        else:
            filename = img_url = image

        # Prepare the evaluation functions with specific system prompts
        eval_func = partial(
            chat, 
            images=[img_url], 
            system_prompt=system_prompt, # Defined in outer scope
            temperature=temperature,     # Defined in outer scope
            top_p=0.9, 
            max_tokens=10, 
            return_logprobs=use_logprobs, # Defined in outer scope
            # extra_body=extra_body,
        )

        semantic_questions = metadata['semantic_questions']
        quality_questions = metadata['quality_questions']

        if isinstance(semantic_questions[0], dict):
            semantic_questions = [meta['question'] for meta in metadata['semantic_questions']]
        
        if isinstance(quality_questions[0], dict):
            quality_questions = [meta['question'] for meta in metadata['quality_questions']]

        # Lists to hold the Future objects
        semantic_futures = []
        quality_futures = []

        # Use a single executor for both task types to maximize concurrency
        # Adjust max_workers as needed (e.g., sum of previous workers or limit based on API rate)
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit semantic tasks
            for q in semantic_questions:
                future = executor.submit(eval_func, q)
                semantic_futures.append(future)

            # Submit quality tasks
            for q in quality_questions:
                future = executor.submit(eval_func, q)
                quality_futures.append(future)
                
            # Gather results (this blocks until each specific task is complete)
            # Using .result() will raise any exceptions that occurred in the threads
            semantic_results = [f.result() for f in semantic_futures]
            quality_results = [f.result() for f in quality_futures]

        # Calculate scores
        sc_score = sum(semantic_results) / len(semantic_results) if semantic_results else 1.0
        pq_score = sum(quality_results) / len(quality_results) if quality_results else 1.0

        if score_method == 'mean':
            o_score = 0.5 * sc_score + 0.5 * pq_score
        elif score_method == 'sum_mean':
            o_score = (sum(semantic_results) + sum(quality_results)) / (len(semantic_results) + len(quality_results))
        else:
            # Calculate overall score (Geometric Mean)
            o_score = math.sqrt(sc_score * pq_score)

        if USE_REMOTE_SERVER:
            delete_remote_image(filename)

        return dict(
            semantic_score=sc_score, 
            semantic_scores=semantic_results,
            quality_score=pq_score, 
            quality_scores=quality_results,
            overall_score=o_score
        )

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = _tensor_to_pil_images(images)

        all_scores = []
        all_metas = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            results_dicts = executor.map(evalue_one_image, images, prompts, metadata)

            for meta in results_dicts:
                all_scores.append(meta['overall_score'])
                all_metas.append(meta)
        return all_scores, all_metas
    return _fn

def decompositional_verifiable_reward_binary(device, backbone="qwen3vl", task='t2i', use_logprobs=False):
    return decompositional_verifiable_reward(device, backbone=backbone, task=task, use_logprobs=use_logprobs, temperature=0.0)

def reflective_think_format_score(device='cuda'):
    import os
    import json
    from concurrent.futures import ThreadPoolExecutor

    judge_system_prompt = """
You are a Strict Logic Compliance Checker. Your ONLY job is to verify if a "Reflection Paragraph" strictly adheres to a 3-step reasoning structure.

You must be rigid and unforgiving. Do not interpret "intent". If the text does not explicitly match the criteria, Score 0.
    
The input text is a single paragraph. You must verify that the text **explicitly** contains the following three steps and DOES NOT contain the forbidden content**.

### THE 3 REQUIRED LOGICAL STEPS:
1.  **Clarification (Anchor):** Does the text start by **explicitly** restating the user's core request/intent? (e.g., "The user wants a cyberpunk city...") -> *Anchors the reasoning.*
2.  **Comparison (Critique):** Does the text **explicitly** identify specific elements in the generated image that fail to meet the request? (e.g., "...but the image lacks neon lights.") -> *Identifies the gap.*
3.  **Solution (Action):** **CRITICAL:** Does the text should conclude with a **EXPLICITLY concrete, actionable instruction** to fix the error? (e.g., "Add neon signage to the buildings.") -> *Directs the fix.* 
  - This step must be an **Imperative Command** telling the generator exactly what to change.
  - **PASS:** "Change the background to a courtyard.", "Add neon lights.", "Remove the shadow." (Action Verbs).
  - **FAIL:** "The image needs to be outdoors.", "The setting is wrong.", "It failed to meet the requirement." (Passive/Descriptive).
  - **FAIL:** Any solution that is merely "implied" by the Critique.

### FORBIDDEN CONTENT (CRITICAL):
1. **No Prompt Expansion:** The text must **NOT** include the prompt expanding content. For example, like "Here's the expand prompt".
2. **Trash Content:** If the text outputs trash content, like repeat "tiesties", direct dict output 0 without any reasoning".

### RESPONSE FORMAT (STRICT JSON):
You must output a valid JSON object containing exactly two keys: `reasoning` and `score`. If the reasoning text contains double quotes, escape them with a backslash (e.g., \")."

- `reasoning`: A brief string analyzing whether each of the 3 steps is present **AND explicitly checking if any "Prompt Expansion" occurred**.
- `score`: Follow the rules below:
    - **Score 1.0:** **ALL 3 steps** are clearly present **AND NO forbidden content (Prompt Expansion) is found**. 
    - **Score 0.0:** If any step is missing **OR if the text tries to expand the prompt**, return `0`.

### EXAMPLE Json OUTPUT:
{
  "reasoning": [Analyze the text step by step to verify the presence of each logical component in the correct order.],
  "score": 0 or 1
}
"""

    client, model_name = _make_openai_client()

    def chat(prompt, system_prompt=None, max_tokens = 1024,
             temperature=0.3, top_p=1.0, **kwargs):
        for _ in range(10):
            try:
                messages = []
                if system_prompt is not None:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({
                    "role": "user",
                    "content": prompt
                })

                if 'extra_headers' in kwargs:
                    kwargs['extra_headers'].update({"X-TT-LOGID": ""})
                else:
                    kwargs['extra_headers'] = {"X-TT-LOGID": ""}

                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                    timeout=180000000,
                )

                completion = completion.json()
                if isinstance(completion, str):
                    completion = json.loads(completion)

                resp = completion['choices'][0]['message']['content']

                return resp
            except Exception as e:
                print(e)
                continue
        print('Warning: Failed to get response after 10 times, return 0.')
        return None
    
    def evalue_one_prompt(prompt):
        parsed_answer = None
        temperature = 0.
        for _ in range(5):
            try:
                answer = chat(prompt, system_prompt=judge_system_prompt, temperature=temperature, top_p=0.8)
                answer = re.sub(r"^```json|```$", "", answer, flags=re.MULTILINE).strip()
                parsed_answer = json.loads(answer)
                parsed_answer['score'] = float(parsed_answer['score'])
                parsed_answer['reasoning'] = parsed_answer['reasoning'].strip()
                break
            except Exception as e:
                print(e, answer, prompt)
                pattern = r'"score"\s*:\s*([\d\.]+)'
                match = re.search(pattern, answer)
                if match:
                    print('Parse json error. Direct parse score value:', answer)
                    parsed_answer = dict(score=float(match.group(1)), reasoning="parse score directly")
                    break
                temperature += 0.2
                
        if parsed_answer is None:
            return 0.0, dict(score=0.0, reasoning="no valid score found")
        return parsed_answer['score'], parsed_answer

    def _fn(prompts):
        all_scores = []
        all_metas = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            results_dicts = executor.map(evalue_one_prompt, prompts)

            for res in results_dicts:
                all_scores.append(res[0])
                all_metas.append(res[1])

        return all_scores, all_metas

    return _fn

def think_tag_format_score():

    def compute_think_text_format_reward(texts):
        """
        Format reward for think text.
        """
        think_start_tag = "<think>"
        think_end_tag = "</think>"
        all_reward = []
        for text in texts:
            starts_correctly = text.startswith(think_start_tag)
            ends_correctly = text.endswith(think_end_tag)

            if starts_correctly and ends_correctly:
                reward = 1.0
            else:
                reward = 0.0

            all_reward.append(reward)
        return all_reward

    def _fn(prompts):
        all_rewards = compute_think_text_format_reward(prompts)
        return all_rewards, {}

    return _fn


def multi_score(device, score_dict):
    from rewards.univr import univr_vrgrpo, univr_vrgrpo_validation

    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "siglipscore": siglip_score,
        "image_similarity": image_similarity_score,
        "hpsv2": hpsv2_score,
        "hpsv3": hpsv3_score_remote,
        "spectrareward": spectrareward_score,
        "viescore_qwen3vl_t2i": viescorer_t2i,
        "viescore_qwen3vl_t2i_compare": viescorer_t2i_compare,
        "dvreward": decompositional_verifiable_reward,
        "dvreward_binary": decompositional_verifiable_reward_binary,
        "reflective_think_format": reflective_think_format_score,
        "think_tag_format": think_tag_format_score,
        "univr_vrgrpo": univr_vrgrpo,
        "univr_vrgrpo_validation": univr_vrgrpo_validation,
    }
    score_fns = {}
    score_weights = {}
    for score_name, spec in score_dict.items():
        if hasattr(spec, "items"):
            weight = float(spec.get("weight", 1.0))
            kwargs = dict(spec.get("kwargs", {}))
        else:
            weight = float(spec)
            kwargs = {}
        factory = score_functions[score_name]
        if "device" in factory.__code__.co_varnames:
            score_fns[score_name] = factory(device=device, **kwargs)
        else:
            score_fns[score_name] = factory(**kwargs)
        score_weights[score_name] = weight

    # only_strict is only for geneval. During training, only the strict reward is needed.
    def _fn(input_data, only_strict=True):
        total_scores = []
        score_metas = {}
        score_details = {}

        for score_name in score_dict:
            weight = score_weights[score_name]
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    **input_data, only_strict=only_strict)
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](**input_data)

            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

            if isinstance(rewards, dict):
                for key, value in rewards.items():
                    if key == "_metadata":
                        score_metas.update(value)
                    else:
                        score_details[key] = value
            elif isinstance(rewards, list) and rewards:
                for reward in rewards:
                    for key, value in reward.items():
                        score_metas.setdefault(key, []).append(value)

        score_details["avg"] = total_scores
        return score_details, score_metas

    return _fn

def debug_image_reward():
    import torchvision.transforms as transforms

    image_paths = [
        "assets/test.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A photo of a cat.',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        # "unifiedreward": 1.0
        # "siglipscore": 0.5,
        "hpsv2": 0.5,
        # "hpsv3": 0.5
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


def debug_text_reward():
    examples = [
        '<think>\nThe user wants a neon-blue katana suspended above a teddy bear, illuminated by a holographic keypad with energy pulsing, and patterned lettering in the background. The original image shows a teddy bear on a keypad with "KAT KAI" glowing, but lacks a katana and lacks energy effects. To refine, add a neon katana above the bear, include pulsing energy effects around the keypad, and introduce patterned lettering like "KAT KAI" in a futuristic style. Keep the clean background and soft lighting.\n</think>',
        '<think>\nThe user requested a classroom with a chalkboard displaying the exact phrase "Field Trip Tomorrow" in colorful letters, along with a map of a nearby nature reserve on the wall to hint at the destination. I correctly rendered the classroom layout, desks, and the chalkboard’s decorative style, but I mistakenly included the word "Tonromony" beneath "Field Trip" and depicted a generic map instead of a nature reserve. This likely stemmed from misinterpreting the destination cue or overcomplicating the text. To fix this, I need to remove the extraneous word "Tonromony" and replace the map with a detailed, colorful map of a nearby nature reserve to accurately fulfill the scene’s educational context.\n</think>',
        "<think>\nThe original image does not reflect the user's request because the snowmen, mice, and setting are inconsistent. The mice are mismatched in size, and the lighting is too bright. To refine the image, the following editing instructions should be followed: \n\n1. Replace the mice with small yellow mice. \n2. Add two small yellow mice to each snowman. \n3. Ensure the mice are positioned correctly, maintaining even spacing. \n4. Add a partially hidden mouse under the leafy canopy at the bottom left. \n5. Adjust the lighting to create softer, more natural illumination. \n6. Retain the grassy field, bar counter, and leafy canopy as these elements align with the user's description.\n</think>",
    ]

    
    score_dict = {
        "reflective_think_format": 1.0,
    }
    device = 'cuda'
    reward_fn = multi_score(device, score_dict)

    print(reward_fn(dict(prompts=examples)))


if __name__ == "__main__":
    debug_image_reward()
    debug_text_reward()
