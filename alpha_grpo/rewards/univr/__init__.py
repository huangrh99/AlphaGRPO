# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""UniVR / VR-GRPO trajectory rewards adapted to BAGEL images."""

from .format_reward import bagel_format_reward
from .reward import (
    UniVRReward, UniVRValidationReward, univr_vrgrpo, univr_vrgrpo_validation,
)

__all__ = [
    "UniVRReward", "UniVRValidationReward", "bagel_format_reward",
    "univr_vrgrpo", "univr_vrgrpo_validation",
]

