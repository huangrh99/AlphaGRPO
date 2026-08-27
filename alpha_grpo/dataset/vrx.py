# Copyright 2026 AlphaGRPO Authors.
# SPDX-License-Identifier: Apache-2.0

"""Native VR-X parquet reader for image-only visual reasoning."""

from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
from torch.utils.data import Dataset


VRX_METADATA_FIELDS = (
    "height",
    "width",
    "sub_tasks",
    "frame_subtask_mapping",
    "vlm_frame_captions",
    "global_summary",
)


def decode_image_bytes(value: Any) -> Image.Image:
    """Decode the byte-like values emitted by common parquet readers."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping):
        for key in ("bytes", "data", "image_bytes"):
            if key in value:
                value = value[key]
                break
    if hasattr(value, "tobytes") and not isinstance(value, (bytes, bytearray, memoryview)):
        value = value.tobytes()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise TypeError(f"Expected encoded image bytes, got {type(value).__name__}")
    with Image.open(io.BytesIO(value)) as image:
        return image.convert("RGB")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (bytes, bytearray, memoryview, Image.Image, Mapping)):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if isinstance(value, (list, tuple)) else [value]


def stable_vrx_uid(dataset_source: str, parquet_file: str, row_index: int, instruction: str) -> str:
    identity = f"{dataset_source}\0{parquet_file}\0{row_index}\0{instruction}".encode("utf-8")
    return f"vrx-{hashlib.sha256(identity).hexdigest()[:24]}"


class VRXDataset(Dataset):
    """Load one or more VR-X sources without importing UniVR at runtime."""

    def __init__(
        self,
        dataset: str | Path | None = None,
        split: str = "train",
        sources: str | Path | Sequence[str | Path] | Mapping[str, str | Path] | None = None,
        max_samples_per_source: int | Mapping[str, int | None] | None = None,
        image_size: int | None = None,
        required_gt_frames: int | None = None,
        seed: int = 42,
    ):
        del split  # VR-X RL sources are per-task parquet files, not split files.
        self.seed = int(seed)
        self.image_size = int(image_size) if image_size is not None else None
        self.required_gt_frames = (
            int(required_gt_frames) if required_gt_frames is not None else None
        )
        self.rows: list[dict[str, Any]] = []
        source_specs = self._normalize_sources(sources if sources is not None else dataset)
        if not source_specs:
            raise ValueError("VRXDataset requires at least one parquet file or source directory")

        for source_name, source_path in source_specs:
            source_rows = self._read_source(source_name, source_path)
            limit = self._source_limit(max_samples_per_source, source_name)
            if limit is not None and limit < len(source_rows):
                rng = random.Random(f"{self.seed}:{source_name}")
                selected = sorted(rng.sample(range(len(source_rows)), max(0, limit)))
                source_rows = [source_rows[index] for index in selected]
            self.rows.extend(source_rows)

    @staticmethod
    def _normalize_sources(sources):
        if sources is None or sources == "":
            return []
        if hasattr(sources, "items"):
            return [(str(name), Path(path).expanduser()) for name, path in sources.items()]
        if isinstance(sources, (str, Path)):
            sources = [sources]
        return [(Path(path).stem, Path(path).expanduser()) for path in sources]

    @staticmethod
    def _source_limit(limits, source_name):
        if limits is None:
            return None
        if hasattr(limits, "items"):
            value = limits.get(source_name)
            return None if value is None else int(value)
        return int(limits)

    def _read_source(self, source_name: str, source_path: Path) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("VR-X parquet loading requires pyarrow") from exc

        if not source_path.exists():
            raise FileNotFoundError(f"VR-X source does not exist: {source_path}")
        files = [source_path] if source_path.is_file() else sorted(source_path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under VR-X source: {source_path}")

        rows = []
        for parquet_path in files:
            for row_index, row in enumerate(pq.read_table(parquet_path).to_pylist()):
                instruction = row.get("vlm_task_instruction") or row.get("question") or row.get("global_summary")
                if instruction is None:
                    raise ValueError(f"Missing VR-X instruction in {parquet_path} row {row_index}")
                instruction = str(instruction)
                relative_file = str(parquet_path.relative_to(source_path)) if source_path.is_dir() else parquet_path.name
                metadata = {key: row.get(key) for key in VRX_METADATA_FIELDS if key in row}
                raw_dataset_source = row.get("dataset_source")
                metadata.update({
                    "uid": stable_vrx_uid(source_name, relative_file, row_index, instruction),
                    "dataset_source": str(raw_dataset_source or source_name),
                    "dataset_partition": source_name,
                    "parquet_file": relative_file,
                    "parquet_row": row_index,
                })
                for key, value in row.items():
                    if key not in {"problem_image_bytes", "answer_image_bytes"} and key not in metadata:
                        metadata[key] = value
                input_images = [
                    decode_image_bytes(value) for value in _as_list(row.get("problem_image_bytes"))
                ]
                gt_trajectory = [
                    decode_image_bytes(value) for value in _as_list(row.get("answer_image_bytes"))
                ]
                if (
                    self.required_gt_frames is not None
                    and len(gt_trajectory) != self.required_gt_frames
                ):
                    continue
                if self.image_size is not None:
                    size = (self.image_size, self.image_size)
                    input_images = [image.resize(size, Image.Resampling.LANCZOS) for image in input_images]
                    gt_trajectory = [image.resize(size, Image.Resampling.LANCZOS) for image in gt_trajectory]
                rows.append({
                    "prompt": instruction,
                    "input_images": input_images,
                    "gt_trajectory": gt_trajectory,
                    "metadata": metadata,
                })
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @staticmethod
    def collate_fn(examples):
        return (
            [example["prompt"] for example in examples],
            [example["metadata"] for example in examples],
            [example["input_images"] for example in examples],
            [example["gt_trajectory"] for example in examples],
        )
