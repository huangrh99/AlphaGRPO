# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible VLM client for CPU/remote UniVR trajectory rewards."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import re
from typing import Iterable

import requests
from PIL import Image

def get_vlm_evaluation_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the evaluation prompt for VLM reward model.
    Aligned with inference_and_evaluate_v3.py: outputs TWO scores (TASK_SCORE, VISUAL_SCORE).

    Returns:
        (system_prompt, user_instruction)
    """
    system_prompt = (
        "You are a reward model for multimodal generative models used in reinforcement learning training. "
        "Your task is to evaluate the quality of the generated images and provide quantitative reward scores.\n"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Logical Coherence**: Can the multiple generated images form a correct reasoning and planning "
        "process, accurately execute the task instruction, and achieve the task goal? Focus on:\n"
        "   - Whether the sequence progresses step by step toward the task goal (e.g., object manipulation, "
        "folding/construction steps, navigation, visual reasoning, etc.).\n"
        "   - Whether the key actions described in the task instruction (grasping, placing, folding, navigating, "
        "locating, assembling, etc.) are correctly performed.\n"
        "   - Whether objects, items, and scene elements maintain consistent quantity, color, and shape throughout the "
        "sequence — no unexpected mutations, appearances, or disappearances.\n"
        "   - Whether physical interactions are plausible and state transitions between frames are smooth and coherent.\n"
        "   - Whether the final state of the sequence matches the expected task outcome.\n"
        "   - If GT reference images are provided, compare whether the generated steps are aligned with the GT steps, "
        "especially whether the final frame matches the GT target state.\n"
        "\n2. **Visual Consistency**: Do the multiple generated images have spatial consistency, with appearance content "
        "(objects, background, etc.) remaining coherent throughout the sequence, without significant abrupt changes or "
        "artifacts, viewable as key frames of continuous actions? Objects and people/robotic arms in the images must not "
        "show obvious distortion or deformation. Evaluate visual coherence and visual quality.\n\n"
        "Output requirements:\n"
        "First provide a brief analysis, then in the **last part** of your response, strictly output two score values between "
        "0.0 and 1.0 (floating point) in the following format:\n"
        "[[TASK_SCORE: 0.xx]]\n"
        "[[VISUAL_SCORE: 0.xx]]\n"
        "Note: Both score lines above must be output; neither can be omitted."
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"

    return system_prompt, user_instruction


def get_task_category(question_text: str = "", dataset_source: str = "") -> str | None:
    """Determine the task category from question text and dataset source."""
    ds = dataset_source.upper() if dataset_source else ""
    q = question_text.lower() if question_text else ""

    # ---------- EgoDex ----------
    if ds == "EGODEX" or "egodex" in ds.lower():
        if "assemble_disassemble_legos" in q or "build_unstack_lego" in q or "assemble_disassemble_soft_legos" in q:
            return "egodex_legos"
        if "basic_fold" in q:
            return "egodex_basic_fold"
        if "fold_unfold_paper" in q:
            return "egodex_fold_paper"
        if "insert_remove_bookshelf" in q:
            return "egodex_bookshelf"
        if "stack_unstack_bowls" in q:
            return "egodex_bowls"
        return "egodex_generic"

    # ---------- VideoCraft ----------
    if ds == "VIDEOCRAFT" or "videocraft" in ds.lower():
        if "airplane" in q or "paper airplane" in q:
            return "craft_airplane"
        if "boat" in q or "paper boat" in q:
            return "craft_boat"
        if "horse" in q:
            return "craft_horse"
        if "tower" in q:
            return "craft_tower"
        if "person" in q:
            return "craft_person"
        return "craft_generic"

    # ---------- Agibot ----------
    if "pickup" in q and "supermarket" in q:
        return "pickup_supermarket"
    if ("produce section" in q or "produce area" in q) and "pickup" in q:
        return "pickup_supermarket"
    if "open" in q and "fridge" in q and ("get food" in q or "take" in q):
        return "open_fridge"
    if "toast bread" in q or ("toast" in q and "bread" in q and "toaster" in q):
        return "toast_bread"
    if "take toast" in q or ("take" in q and "toaster" in q):
        return "take_toast"
    if "sort" in q and "warehouse" in q:
        return "sort_warehouse"
    if "fold" in q or "unfold" in q or "flatten" in q or "stack" in q:
        return "folding"
    if "clear" in q and ("countertop" in q or "counter" in q) and "waste" in q:
        return "clear_countertop"
    if ("dispose" in q or "trash" in q or "waste" in q or "garbage" in q) and ("desk" in q or "table" in q or "countertop" in q):
        return "clear_countertop"
    if "boil water" in q and "kettle" in q:
        return "boil_water"
    if "open" in q and "red wine" in q:
        return "open_wine"
    if "inner pot" in q and "rice cooker" in q:
        return "rice_cooker"
    if "rice" in q and "cooker" in q and "pot" in q:
        return "rice_cooker"
    if "wardrobe" in q and "hang" in q and "clothes" in q:
        return "hang_wardrobe"
    if "open" in q and "wardrobe" in q and "hang" in q:
        return "hang_wardrobe"
    if "swipe" in q and ("card" in q or "toy" in q):
        return "swipe_cards"
    if "hang" in q and "hanger" in q:
        return "hang_hanger"

    return None


_TASK_CRITERIA: dict[str, str] = {
    "pickup_supermarket": (
        "\n**Task Background**: This is a robot task for picking up items in a supermarket.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm successfully grasps the specified item "
        "and places it correctly into the shopping cart at the bottom of the image. If the item is not grasped, the wrong item is "
        "grasped, or it is not placed in the cart, the task completion score should be significantly reduced.\n"
    ),
    "open_fridge": (
        "\n**Task Background**: This is a robot task for taking food out of a refrigerator.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm correctly retrieves items from the "
        "refrigerator and places them on the table with the same quantity and shape as the GT. If the quantity is inconsistent, "
        "the shape has changed significantly, or the item was not successfully retrieved, the score should be significantly reduced.\n"
    ),
    "toast_bread": (
        "\n**Task Background**: This is a robot task for toasting bread.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm correctly places bread slices into "
        "the toaster. If bread slices do not enter the toaster or the placement position is clearly wrong, the score should be reduced.\n"
    ),
    "take_toast": (
        "\n**Task Background**: This is a robot task for taking toast slices out of the toaster.\n"
        "**Task Completion Evaluation Focus**: Pay close attention to whether the robotic arm takes both toast slices out of the "
        "toaster. If the quantity is insufficient or retrieval fails, the score should be reduced.\n"
    ),
    "sort_warehouse": (
        "\n**Task Background**: This is a robot warehouse sorting task.\n"
        "**Task Completion Evaluation Focus**: Determine whether the target item has been correctly moved to the specified location. "
        "If the item does not reach the target location or is placed in the wrong location, the score should be reduced.\n"
    ),
    "folding": (
        "\n**Task Background**: This is a robot task involving folding/unfolding clothes.\n"
        "**Task Completion Evaluation Focus**: Focus on the shape change process of the clothing. It is not required that the "
        "folding process be exactly the same as the GT; as long as there is no obvious distortion and the clothing is shown "
        "being folded or unfolded step by step toward the target state, it is acceptable. The final shape should be basically "
        "consistent with the GT target state. If the clothing shows severe distortion or the final shape deviates too far from "
        "the GT, the score should be reduced.\n"
    ),
    "clear_countertop": (
        "\n**Task Background**: This is a robot task for clearing waste from a countertop.\n"
        "**Task Completion Evaluation Focus**: Focus on whether all items on the countertop have been placed into the trash bin "
        "and whether the trash bin lid is finally closed. If items are missed or the trash bin is not closed, the score should be reduced.\n"
    ),
    "boil_water": (
        "\n**Task Background**: This is a robot task for boiling water with a kettle.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the kettle is placed under the faucet to fill with water, whether "
        "the lid is closed after it is full, and whether it is placed back in its original position. If any step is missing "
        "(e.g., no water collected, lid not closed, not returned to original position), the score should be reduced.\n"
    ),
    "open_wine": (
        "\n**Task Background**: This is a robot task for opening red wine.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the wine bottle is aligned with the decanter opening and whether "
        "the wine successfully fills the decanter. If the bottle is not aligned or the wine is not poured in, the score should be reduced.\n"
    ),
    "rice_cooker": (
        "\n**Task Background**: This is a robot task for placing the inner pot into a rice cooker.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the pot is placed into the rice cooker by the robotic arm and "
        "whether the robotic arm presses the button to start cooking. If the pot is not placed in or the button is not pressed, "
        "the score should be reduced.\n"
    ),
    "hang_wardrobe": (
        "\n**Task Background**: This is a robot task for opening a wardrobe and hanging clothes.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the clothes held by the robotic arm are correctly hung inside the "
        "wardrobe and whether the wardrobe door is finally closed. If the clothes are not properly hung, the score should be reduced.\n"
    ),
    "swipe_cards": (
        "\n**Task Background**: This is a robot task for swiping toy cards.\n"
        "**Task Completion Evaluation Focus**: Focus on whether the robotic arm grasps the white card and, while holding it, "
        "touches the card-swiping position. If the card is not grasped or the swiping position is not touched, the score should be reduced.\n"
    ),
    "hang_hanger": (
        "\n**Task Background**: This is a robot task for hanging clothes with a hanger.\n"
        "**Task Completion Evaluation Focus**: It is not required to strictly follow the GT hanging steps, but the hanger should "
        "be attached to the clothes step by step and the final shape should be similar to the GT. If the final shape deviates "
        "too far from the GT or there are serious anomalies during the process, the score should be reduced.\n"
    ),
    "egodex_legos": (
        "\n**Task Background**: This is a task involving assembly or disassembly of LEGO/soft LEGO bricks (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on whether the color and quantity of bricks have mutated. Even if assembly "
        "or disassembly is completed, if the quantity and color of bricks change (appear out of nowhere, disappear, or change "
        "color), a low score should be given. It is not required that the final shape be exactly the same as the GT; different "
        "construction methods that still meet the task instruction requirements are acceptable.\n"
    ),
    "egodex_basic_fold": (
        "\n**Task Background**: This is a basic folding task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on the shape change process of the target object. It is not required that "
        "the folding process be exactly the same as the GT; as long as there is no obvious distortion and the target object is "
        "shown being folded or unfolded step by step toward the target state, it is acceptable.\n"
    ),
    "egodex_fold_paper": (
        "\n**Task Background**: This is a paper folding/unfolding task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on whether the paper deformation is reasonable, whether the creases are "
        "clear, whether the paper shape change is continuous, and whether it is progressing toward the target state step by step. "
        "If the paper shows unreasonable deformation, tearing, or deviates too far from the target state, the score should be reduced.\n"
    ),
    "egodex_bookshelf": (
        "\n**Task Background**: This is a task for inserting or removing books from a bookshelf (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Focus on whether the books are correctly placed into or taken out of the bookshelf, "
        "while the quantity and shape of the books must not change (cannot appear out of nowhere, disappear, or change shape).\n"
    ),
    "egodex_bowls": (
        "\n**Task Background**: This is a bowl stacking or unstacking task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: While judging whether the task instruction is completed, focus on whether the "
        "quantity and color of the bowls have mutated (cannot appear out of nowhere, disappear, or change color).\n"
    ),
    "egodex_generic": (
        "\n**Task Background**: This is a tabletop manipulation task (hand-operated).\n"
        "**Task Completion Evaluation Focus**: Determine whether the generated image sequence correctly executes the task "
        "instruction; the quantity, color, and shape of items during the operation should remain reasonable without obvious mutations.\n"
    ),
    "craft_airplane": (
        "\n**Task Background**: This is a paper airplane folding task. "
        "The reference step images show the correct sequence of paper airplane folding steps. "
        "Compare whether the generated image sequence follows the correct folding steps and order.\n"
    ),
    "craft_boat": (
        "\n**Task Background**: This is a paper boat folding task. "
        "The reference step images show the correct sequence of paper boat folding steps. "
        "Compare whether the generated image sequence follows the correct folding steps and order.\n"
    ),
    "craft_horse": (
        "\n**Task Background**: This is a task for building a horse with building blocks. "
        "The reference step images show the correct sequence of building steps. "
        "Compare whether the generated image sequence follows the correct building steps and order.\n"
    ),
    "craft_tower": (
        "\n**Task Background**: This is a task for building a tower with building blocks. "
        "The reference step images show the correct sequence of building steps. "
        "Compare whether the generated image sequence follows the correct building steps and order.\n"
    ),
    "craft_person": (
        "\n**Task Background**: This is a task for building a person figure with building blocks. "
        "The reference step images show the correct sequence of building steps. "
        "Compare whether the generated image sequence follows the correct building steps and order.\n"
    ),
    "craft_generic": (
        "\n**Task Background**: This is a handicraft making task. "
        "Compare whether the generated image sequence follows the correct making steps and order.\n"
    ),
}


def get_vlm_validation_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the evaluation prompt for VLM reward model.
    Aligned with inference_and_evaluate_v3.py: dual-score (TASK_SCORE + VISUAL_SCORE).
    
    Args:
        question_text: The task instruction/question
        dataset_source: Dataset source identifier (e.g. "agibot", "egodex", "videocraft")
    
    Returns:
        (system_prompt, user_instruction)
    """
    category = get_task_category(question_text, dataset_source)
    task_instruction = _TASK_CRITERIA.get(category, "") if category else ""

    # Determine if this is an Agibot or EgoDex task (has specific criteria)
    _AGIBOT_CATEGORIES = {
        "pickup_supermarket", "open_fridge", "toast_bread", "take_toast",
        "sort_warehouse", "folding", "clear_countertop", "boil_water",
        "open_wine", "rice_cooker", "hang_wardrobe", "swipe_cards", "hang_hanger",
    }
    _EGODEX_CATEGORIES = {
        "egodex_legos", "egodex_basic_fold", "egodex_fold_paper",
        "egodex_bookshelf", "egodex_bowls", "egodex_generic",
    }
    has_specific_criteria = (
        (category is not None and category in _AGIBOT_CATEGORIES) or
        (category is not None and category in _EGODEX_CATEGORIES)
    )

    system_prompt = (
        "You are a reward model for multimodal generative models used in reinforcement learning training. "
        "Your task is to evaluate the quality of the generated images and provide quantitative reward scores.\n"
        f"{task_instruction}"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Logical Coherence**: Can the multiple generated images form a correct reasoning and planning "
        "process, accurately execute the task instruction, and help achieve the task goal? "
        "If GT reference images are provided, focus on comparing whether the generated steps are consistent with the GT steps."
    )
    if has_specific_criteria:
        system_prompt += " Please pay special attention to the evaluation focus described in the task background above."
    else:
        system_prompt += (
            " For origami tasks, focus on whether the paper shape changes follow the sequence shown in the reference steps, "
            "and whether the shape and fold marks of the paper at each step are similar to the reference images."
        )
    system_prompt += (
        "\n2. **Visual Consistency**: Do the multiple generated images have spatial consistency, with appearance content "
        "(objects, background, etc.) remaining coherent throughout the sequence, without significant abrupt changes or "
        "artifacts, viewable as key frames of continuous actions? Objects and people/robotic arms in the images must not "
        "show obvious distortion or deformation. Evaluate visual coherence and visual quality.\n\n"
        "Output requirements:\n"
        "First provide a brief analysis, then in the **last part** of your response, strictly output two score values between "
        "0.0 and 1.0 (floating point) in the following format:\n"
        "[[TASK_SCORE: 0.xx]]\n"
        "[[VISUAL_SCORE: 0.xx]]\n"
        "Note: Both score lines above must be output; neither can be omitted."
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"
    
    return system_prompt, user_instruction


def get_vlm_sampled_frame_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the evaluation prompt for comparing a single generated frame
    with the corresponding GT frame using VLM.

    Aligned with v3: uses the same 2-criterion format (TASK_SCORE + VISUAL_SCORE).

    Returns:
        (system_prompt, user_instruction)
    """
    system_prompt = (
        "You are an image similarity evaluation model for reward computation in reinforcement learning training.\n"
        "Your task is to compare the [Generated Frame] with the [Corresponding Ground Truth Frame] "
        "and evaluate whether the generated result successfully achieves the task goal state.\n"
        "Evaluation criteria (please score each separately):\n"
        "1. **Task Completion and Goal Consistency**: Does the generated frame present the same goal state as the "
        "GT corresponding frame? Focus on:\n"
        "   - Whether key task elements (object positions, poses, operation progress, spatial relationships, etc.) "
        "match the GT frame.\n"
        "   - Whether objects, items, and scene elements maintain consistent quantity, color, and shape compared to "
        "the GT — no unexpected mutations, appearances, or disappearances.\n"
        "   - Whether the depicted action or state (grasping, placing, folding, navigating, assembling, etc.) "
        "is consistent with the task instruction and the GT goal state.\n"
        "   - Whether physical plausibility is maintained (no impossible poses, coherent interactions).\n"
        "\n2. **Visual Consistency**: The visual quality of the generated frame — whether the shape, color, and size "
        "of objects are consistent with the GT, and whether there are artifacts, distortions, or warping.\n\n"
        "Output requirements:\n"
        "First provide a brief comparative analysis, then in the **last part** of your response, strictly output two "
        "score values between 0.0 and 1.0 (floating point) in the following format:\n"
        "[[TASK_SCORE: 0.xx]]\n"
        "[[VISUAL_SCORE: 0.xx]]\n"
        "Note: Both score lines above must be output; neither can be omitted."
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"

    return system_prompt, user_instruction


def get_vlm_pairwise_prompt(question_text: str, dataset_source: str = "") -> tuple[str, str]:
    """
    Construct the pairwise comparison prompt for PPRM (Pairwise Preference Reward Model).

    Instead of scoring a single image sequence absolutely, we compare two image sequences
    and ask the VLM to pick the better one. This produces more stable reward signals.

    Returns:
        (system_prompt, user_instruction)
    """
    system_prompt = (
        "You are a Pairwise Preference Reward Model (PPRM) for comparative evaluation of image generation quality "
        "in reinforcement learning training.\n"
        "Your task is to compare two image sequences generated from the same task instruction and determine which is better.\n"
        "You may receive a Ground Truth reference image sequence representing the ideal output for this task; "
        "please use it as an important reference for your judgment.\n"
        "Evaluation criteria:\n"
        "1. **Task Completion and Logical Coherence**: Which image sequence more accurately executes the task instruction "
        "and forms a more reasonable reasoning and planning process? Focus on:\n"
        "   - Which sequence better progresses step by step toward the task goal (object manipulation, "
        "folding/construction steps, navigation, visual reasoning, etc.).\n"
        "   - Which sequence correctly performs the key actions described in the task (grasping, placing, folding, "
        "navigating, locating, assembling, etc.).\n"
        "   - Which sequence better maintains consistent quantity, color, and shape of objects and scene elements "
        "throughout — fewer unexpected mutations, appearances, or disappearances.\n"
        "   - Which sequence shows more plausible physical interactions and smoother state transitions between frames.\n"
        "   - Which sequence's final state is closer to the expected task outcome.\n"
        "   - If GT reference images are provided, compare whether the generated steps are aligned with the GT steps, "
        "especially whether the last frame is similar to the last frame of the Ground Truth.\n"
        "\n2. **Visual Consistency**: Which image sequence is better in terms of spatial consistency and appearance coherence, "
        "with fewer artifacts, and without obvious distortion or deformation of objects and people/robotic arms.\n\n"
        "Output requirements:\n"
        "First provide a brief comparative analysis, then on the **last line** of your response, strictly output your choice "
        "in the following format:\n"
        "- If Sequence A is better, output: [[PREFER: A]]\n"
        "- If Sequence B is better, output: [[PREFER: B]]\n"
        "- If both are roughly equal in quality, output: [[PREFER: TIE]]\n"
    )

    user_instruction = f"Task Instruction (User Question):\n{question_text}"

    return system_prompt, user_instruction




def _encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_parts(label: str, images: Iterable[Image.Image]):
    parts = [{"type": "text", "text": label}]
    for image in images:
        parts.append({"type": "image_url", "image_url": {"url": _encode_image(image)}})
    return parts



class OpenAICompatibleVLMClient:
    def __init__(
        self,
        api_url=None,
        model=None,
        api_key=None,
        timeout=200.0,
        pairwise_timeout=2000.0,
        validation_prompt=False,
    ):
        self.api_url = (api_url or os.environ.get("UNIVR_REWARD_API_URL") or "").rstrip("/")
        self.model = model or os.environ.get("UNIVR_REWARD_MODEL") or "qwen3vl"
        self.api_key = api_key or os.environ.get("UNIVR_REWARD_API_KEY") or ""
        self.timeout = float(timeout)
        self.pairwise_timeout = float(pairwise_timeout)
        self.validation_prompt = bool(validation_prompt)
        if not self.api_url:
            raise ValueError("Set UNIVR_REWARD_API_URL or configure vrx.reward.api_url")
        if not self.api_url.endswith("/chat/completions"):
            self.api_url += "/chat/completions"

    def _complete(self, content, timeout):
        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.1,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": content}],
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    @staticmethod
    def _score_from_text(text):
        task = re.search(r"\[\[TASK_SCORE:\s*([0-9.]+)\]\]", text or "")
        visual = re.search(r"\[\[VISUAL_SCORE:\s*([0-9.]+)\]\]", text or "")
        single = re.search(r"\[\[SCORE:\s*([0-9.]+)\]\]", text or "")
        values = []
        for match in (task, visual):
            if match:
                try:
                    values.append(max(0.0, min(1.0, float(match.group(1)))))
                except ValueError:
                    pass
        if values:
            return sum(values) / len(values)
        if single:
            try:
                return max(0.0, min(1.0, float(single.group(1))))
            except ValueError:
                pass
        return 0.3

    def score_trajectory(
        self, instruction, input_images, generated_images, gt_images, metadata=None
    ):
        if not generated_images:
            return 0.0
        dataset_source = (metadata or {}).get("dataset_source", "")
        prompt_fn = (
            get_vlm_validation_prompt
            if self.validation_prompt
            else get_vlm_evaluation_prompt
        )
        system_text, instruction_text = prompt_fn(str(instruction), dataset_source)
        content = [{"type": "text", "text": system_text + "\n\n" + instruction_text}]
        if gt_images:
            content += _image_parts(
                "\n\nBelow is the Ground Truth answer image sequence for this sample "
                "(GT Answer Image Sequence):",
                gt_images,
            )
        if input_images:
            content += _image_parts(
                "\n\nBelow are the reference images from the model input "
                "(Input Reference Images):",
                input_images,
            )
        content += _image_parts(
            "\n\nBelow is the model-generated image sequence to be evaluated "
            "(Model Generated Sequence):",
            generated_images,
        )
        content.append({
            "type": "text",
            "text": "\n\nPlease provide a detailed evaluation of the generated image "
            "sequence according to the above criteria:",
        })
        text = self._complete(content, self.timeout)
        return 0.5 if text is None else self._score_from_text(text)

    def compare_trajectories(
        self, instruction, input_images, images_a, images_b, gt_images, metadata=None
    ):
        if not images_a or not images_b:
            if images_a:
                return "A"
            if images_b:
                return "B"
            return "TIE"
        dataset_source = (metadata or {}).get("dataset_source", "")
        system_text, instruction_text = get_vlm_pairwise_prompt(
            str(instruction), dataset_source
        )
        content = [{"type": "text", "text": system_text + "\n\n" + instruction_text}]
        if gt_images:
            content += _image_parts(
                "\n\nBelow is the Ground Truth answer image sequence for this sample "
                "(GT Answer Image Sequence):",
                gt_images,
            )
        if input_images:
            content += _image_parts(
                "\n\nBelow are the reference images from the model input "
                "(Input Reference Images):",
                input_images,
            )
        content += _image_parts(
            "\n\nBelow are the images generated by Sequence A (Sequence A):", images_a
        )
        content += _image_parts(
            "\n\nBelow are the images generated by Sequence B (Sequence B):", images_b
        )
        content.append({
            "type": "text",
            "text": "\n\nPlease compare the two image sequences and determine which is better:",
        })
        text = self._complete(content, self.pairwise_timeout)
        if text is None:
            return "TIE"
        match = re.search(r"\[\[PREFER:\s*(A|B|TIE)\]\]", text, re.IGNORECASE)
        return match.group(1).upper() if match else "TIE"

    def score_frame(
        self,
        instruction,
        input_images,
        generated_image,
        gt_image,
        frame_index,
        metadata=None,
    ):
        del frame_index
        if generated_image is None:
            return 0.0
        if gt_image is None:
            return 0.5
        dataset_source = (metadata or {}).get("dataset_source", "")
        system_text, instruction_text = get_vlm_sampled_frame_prompt(
            str(instruction), dataset_source
        )
        content = [{"type": "text", "text": system_text + "\n\n" + instruction_text}]
        content += _image_parts(
            "\n\nBelow is the Ground Truth corresponding frame for this sample "
            "(representing the task goal state):",
            [gt_image],
        )
        if input_images:
            content += _image_parts(
                "\n\nBelow are the reference images from the model input "
                "(Input Reference Images):",
                input_images,
            )
        content += _image_parts(
            "\n\nBelow is the model-generated corresponding frame (to be evaluated):",
            [generated_image],
        )
        content.append({
            "type": "text",
            "text": "\n\nPlease compare the generated frame with the GT corresponding "
            "frame and evaluate the degree of goal state matching:",
        })
        text = self._complete(content, self.timeout)
        return 0.5 if text is None else self._score_from_text(text)


class MockVLMClient:
    """Deterministic offline client for tests and plumbing smoke runs."""

    @staticmethod
    def _score(images):
        digest = hashlib.sha256()
        for image in images:
            digest.update(image.convert("RGB").resize((8, 8)).tobytes())
        return int.from_bytes(digest.digest()[:2], "big") / 65535.0

    def score_trajectory(self, instruction, input_images, generated_images, gt_images, metadata=None):
        return self._score(generated_images)

    def compare_trajectories(self, instruction, input_images, images_a, images_b, gt_images, metadata=None):
        a, b = self._score(images_a), self._score(images_b)
        return "A" if a > b else "B" if b > a else "TIE"

    def score_frame(self, instruction, input_images, generated_image, gt_image, frame_index, metadata=None):
        return self._score([generated_image])

