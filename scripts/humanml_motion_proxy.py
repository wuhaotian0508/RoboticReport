"""HumanML3D motion-derived proxy metrics and candidate selection.

The parquet HumanML3D export stores processed 263-D motion features. These are
not MoConVQ discrete tokens, but they still carry useful kinematic information.
This module extracts stable proxy metrics from those features and uses them to
rank teacher-generated candidates before LoRA distillation caches are written.
"""

from __future__ import annotations

import math
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SELECTION_METRICS = (
    "duration_sec",
    "root_speed_mean",
    "root_path_length",
    "mean_joint_velocity",
    "mean_joint_acceleration",
    "mean_joint_jerk",
    "root_vertical_range",
    "all_joint_vertical_range",
)


def read_style_rows(csv_path: Path, max_samples: int | None, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            caption = (row.get("caption") or "").strip()
            if not caption:
                continue
            rows.append(
                {
                    "caption": caption,
                    "motion_id": (row.get("motion_id") or "").strip(),
                    "text_path": (row.get("text_path") or "").strip(),
                    "style_groups": (row.get("style_groups") or "").strip(),
                    "matched_keywords": (row.get("matched_keywords") or "").strip(),
                }
            )
    random.Random(seed).shuffle(rows)
    if max_samples:
        rows = rows[:max_samples]
    return rows


def motion_to_array(motion: Any) -> np.ndarray:
    array = np.asarray(motion, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D HumanML3D motion array, got shape {array.shape}")
    if array.shape[0] < 1 or array.shape[1] < 4:
        raise ValueError(f"HumanML3D motion array is too small: {array.shape}")
    return array


def _safe_mean_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=-1).mean())


def _humanml_vertical_columns(width: int) -> list[int]:
    columns = [3]
    # Standard HumanML3D features keep 21 relative joint XYZ triples after the
    # first four root features. The Y values are columns 5, 8, ..., 65.
    columns.extend(range(5, min(width, 67), 3))
    return [col for col in columns if col < width]


def compute_humanml_metrics(motion: Any, fps: float = 20.0) -> dict[str, float]:
    array = motion_to_array(motion)
    frame_time = 1.0 / fps
    frames = array.shape[0]

    root_velocity = array[:, 1:3]
    root_height = array[:, 3]
    vertical = array[:, _humanml_vertical_columns(array.shape[1])]
    feature_delta = np.diff(array, axis=0)
    feature_velocity = feature_delta / frame_time if len(feature_delta) else feature_delta
    feature_acceleration = np.diff(feature_velocity, axis=0) / frame_time if len(feature_velocity) > 1 else np.empty((0, array.shape[1]))
    feature_jerk = np.diff(feature_acceleration, axis=0) / frame_time if len(feature_acceleration) > 1 else np.empty((0, array.shape[1]))

    root_speed = np.linalg.norm(root_velocity, axis=-1)
    return {
        "frames": float(frames),
        "fps": float(fps),
        "duration_sec": float(frames * frame_time),
        "root_speed_mean": float(root_speed.mean()) if root_speed.size else 0.0,
        "root_path_length": float(root_speed.sum() * frame_time),
        "mean_joint_velocity": _safe_mean_norm(feature_velocity),
        "mean_joint_acceleration": _safe_mean_norm(feature_acceleration),
        "mean_joint_jerk": _safe_mean_norm(feature_jerk),
        "root_vertical_range": float(np.ptp(root_height)) if root_height.size else 0.0,
        "root_height_mean": float(root_height.mean()) if root_height.size else 0.0,
        "all_joint_vertical_range": float(np.ptp(vertical)) if vertical.size else 0.0,
    }


def metric_distance(candidate: dict[str, float], target: dict[str, float], key: str) -> float:
    if key not in candidate or key not in target:
        return 0.0
    c = max(float(candidate[key]), 0.0)
    t = max(float(target[key]), 0.0)
    return abs(math.log((c + 1e-6) / (t + 1e-6)))


def score_candidate_metrics(
    candidate_metrics: dict[str, float],
    target_metrics: dict[str, float],
    metric_names: tuple[str, ...] = DEFAULT_SELECTION_METRICS,
) -> float:
    distances = [
        metric_distance(candidate_metrics, target_metrics, key)
        for key in metric_names
        if key in candidate_metrics and key in target_metrics
    ]
    if not distances:
        return 0.0
    return float(sum(distances) / len(distances))


def select_best_candidate(
    candidates: list[dict[str, Any]],
    target_metrics: dict[str, float],
    metric_names: tuple[str, ...] = DEFAULT_SELECTION_METRICS,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No candidates to select from")
    for candidate in candidates:
        metrics = candidate.get("metrics") or {}
        candidate["selection_score"] = score_candidate_metrics(metrics, target_metrics, metric_names)
    return min(candidates, key=lambda item: float(item["selection_score"]))


class HumanMLMotionStore:
    """Lazy parquet reader keyed by file path and HumanML3D motion id."""

    def __init__(self) -> None:
        self._cache: dict[Path, dict[str, np.ndarray]] = {}

    def load_motion(self, parquet_path: Path, motion_id: str) -> np.ndarray:
        parquet_path = Path(parquet_path)
        motion_id = str(motion_id)
        if parquet_path not in self._cache:
            self._cache[parquet_path] = self._read_file(parquet_path)
        try:
            return self._cache[parquet_path][motion_id]
        except KeyError as exc:
            raise KeyError(f"Motion id {motion_id!r} not found in {parquet_path}") from exc

    def metrics_for(self, parquet_path: Path, motion_id: str, fps: float = 20.0) -> dict[str, float]:
        return compute_humanml_metrics(self.load_motion(parquet_path, motion_id), fps=fps)

    @staticmethod
    def _read_file(parquet_path: Path) -> dict[str, np.ndarray]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read HumanML3D parquet motion data") from exc

        parquet_file = pq.ParquetFile(parquet_path)
        motions: dict[str, np.ndarray] = {}
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index, columns=["motion", "meta_data"])
            for row_index, item in enumerate(table.to_pylist()):
                meta = item.get("meta_data") or {}
                name = str(meta.get("name") or f"{parquet_path.stem}_{row_group_index}_{row_index}")
                motions[name] = motion_to_array(item["motion"])
        return motions
