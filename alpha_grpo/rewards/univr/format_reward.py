# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""BAGEL-native structural equivalent of UniVR's Emu token format reward."""

from __future__ import annotations

from PIL import Image


def bagel_format_reward(
    trajectory,
    expected_num_frames: int | None = None,
    expected_resolution: tuple[int, int] | None = None,
):
    images = list(trajectory or [])
    details = {
        "num_images": len(images),
        "expected_num_images": expected_num_frames,
        "valid_images": 0,
        "dimension_consistent": True,
    }
    if not images:
        details["error"] = "missing trajectory"
        return 0.0, details
    if expected_num_frames is not None and len(images) != expected_num_frames:
        details["error"] = "wrong image count"
        return 0.0, details

    sizes = set()
    for image in images:
        if not isinstance(image, Image.Image):
            details["error"] = f"invalid frame type: {type(image).__name__}"
            return 0.0, details
        if image.width <= 0 or image.height <= 0:
            details["error"] = "empty frame"
            return 0.0, details
        rgb = image.convert("RGB")
        details["valid_images"] += 1
        sizes.add(rgb.size)
        if expected_resolution is not None and rgb.size != tuple(expected_resolution):
            details["error"] = f"unexpected resolution: {rgb.size}"
            return 0.0, details

    if len(sizes) != 1:
        details["dimension_consistent"] = False
        details["error"] = f"inconsistent dimensions: {sorted(sizes)}"
        return 0.0, details
    return 1.0, details

