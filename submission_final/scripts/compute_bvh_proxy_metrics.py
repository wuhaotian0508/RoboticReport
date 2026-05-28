"""Compute BVH-derived proxy metrics for style comparison.

These metrics are not official FID/R-Precision. They quantify observable
kinematic properties of generated BVH files so baseline and LoRA outputs can be
compared on held-out prompts with the same parser and units.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Joint:
    name: str
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    children: list["Joint"] = field(default_factory=list)
    parent: "Joint | None" = None


def rot_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


ROT = {"Xrotation": rot_x, "Yrotation": rot_y, "Zrotation": rot_z}
POS_AXIS = {"Xposition": 0, "Yposition": 1, "Zposition": 2}


def parse_bvh(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    idx = 0
    channel_joints: list[Joint] = []
    all_joints: list[Joint] = []
    end_count = 0

    def parse_joint(parent: Joint | None = None) -> Joint:
        nonlocal idx, end_count
        header = lines[idx]
        if header.startswith("ROOT ") or header.startswith("JOINT "):
            name = header.split(maxsplit=1)[1]
        elif header.startswith("End Site"):
            end_count += 1
            name = f"{parent.name}_end{end_count}" if parent else f"end{end_count}"
        else:
            raise ValueError(f"Unexpected BVH line {idx}: {header}")
        idx += 1
        if lines[idx] != "{":
            raise ValueError(f"Expected '{{' after {name}")
        idx += 1

        joint = Joint(
            name=name,
            offset=np.zeros(3, dtype=np.float64),
            channels=[],
            children=[],
            parent=parent,
        )
        all_joints.append(joint)

        while idx < len(lines):
            line = lines[idx]
            if line.startswith("OFFSET"):
                joint.offset = np.array([float(x) for x in line.split()[1:4]], dtype=np.float64)
                idx += 1
            elif line.startswith("CHANNELS"):
                parts = line.split()
                count = int(parts[1])
                joint.channels = parts[2 : 2 + count]
                channel_joints.append(joint)
                idx += 1
            elif line.startswith("JOINT ") or line.startswith("End Site"):
                child = parse_joint(joint)
                joint.children.append(child)
            elif line == "}":
                idx += 1
                return joint
            else:
                raise ValueError(f"Unexpected BVH line {idx}: {line}")
        raise ValueError("Unexpected end of hierarchy")

    if lines[idx] != "HIERARCHY":
        raise ValueError("Not a BVH file: missing HIERARCHY")
    idx += 1
    root = parse_joint(None)
    while idx < len(lines) and lines[idx] != "MOTION":
        idx += 1
    if idx >= len(lines):
        raise ValueError("Missing MOTION section")
    idx += 1
    frames = int(lines[idx].split(":", 1)[1].strip())
    idx += 1
    frame_time = float(lines[idx].split(":", 1)[1].strip())
    idx += 1
    motion = np.array([[float(x) for x in line.split()] for line in lines[idx : idx + frames]], dtype=np.float64)
    return root, channel_joints, all_joints, motion, frame_time


def frame_pose(root: Joint, frame: np.ndarray) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    cursor = 0

    def visit(joint: Joint, parent_pos: np.ndarray, parent_rot: np.ndarray):
        nonlocal cursor
        local_offset = joint.offset.copy()
        local_rot = np.eye(3)
        for channel in joint.channels:
            value = frame[cursor]
            cursor += 1
            if channel in POS_AXIS:
                local_offset[POS_AXIS[channel]] += value
            elif channel in ROT:
                local_rot = local_rot @ ROT[channel](value)
        global_pos = parent_pos + parent_rot @ local_offset
        global_rot = parent_rot @ local_rot
        positions[joint.name] = global_pos
        for child in joint.children:
            visit(child, global_pos, global_rot)

    visit(root, np.zeros(3), np.eye(3))
    return positions


def stack_positions(root: Joint, all_joints: list[Joint], motion: np.ndarray) -> tuple[list[str], np.ndarray]:
    names = [joint.name for joint in all_joints]
    frames = []
    for frame in motion:
        pose = frame_pose(root, frame)
        frames.append(np.stack([pose[name] for name in names], axis=0))
    return names, np.stack(frames, axis=0)


def safe_mean_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=-1).mean())


def root_y_rotation(channel_joints: list[Joint], motion: np.ndarray) -> np.ndarray:
    cursor = 0
    for joint in channel_joints:
        width = len(joint.channels)
        if joint.parent is None:
            for local_idx, channel in enumerate(joint.channels):
                if channel == "Yrotation":
                    return motion[:, cursor + local_idx]
        cursor += width
    return np.zeros(motion.shape[0], dtype=np.float64)


def compute_metrics(path: Path) -> dict[str, float | str]:
    root, channel_joints, all_joints, motion, frame_time = parse_bvh(path)
    names, pos = stack_positions(root, all_joints, motion)
    fps = 1.0 / frame_time
    duration = len(motion) * frame_time
    root_pos = pos[:, 0, :]
    root_horizontal = root_pos[:, [0, 2]]
    root_delta = np.diff(root_horizontal, axis=0)
    joint_vel = np.diff(pos, axis=0) / frame_time
    joint_acc = np.diff(joint_vel, axis=0) / frame_time
    joint_jerk = np.diff(joint_acc, axis=0) / frame_time

    arm_indices = [
        i
        for i, name in enumerate(names)
        if any(part in name.lower() for part in ("shoulder", "elbow", "wrist", "clavicle"))
    ]
    arm = pos[:, arm_indices, :] if arm_indices else pos[:, :0, :]
    arm_centered = arm - root_pos[:, None, :] if arm_indices else arm
    arm_range = float(np.ptp(arm_centered.reshape(-1, 3), axis=0).sum()) if arm_indices else 0.0

    yaw = root_y_rotation(channel_joints, motion)
    metrics: dict[str, float | str] = {
        "file": str(path),
        "frames": float(len(motion)),
        "fps": float(fps),
        "duration_sec": float(duration),
        "root_speed_mean": safe_mean_norm(root_delta / frame_time),
        "root_path_length": float(np.linalg.norm(root_delta, axis=-1).sum()),
        "mean_joint_velocity": safe_mean_norm(joint_vel),
        "mean_joint_acceleration": safe_mean_norm(joint_acc),
        "mean_joint_jerk": safe_mean_norm(joint_jerk),
        "root_vertical_range": float(root_pos[:, 1].max() - root_pos[:, 1].min()),
        "root_height_mean": float(root_pos[:, 1].mean()),
        "all_joint_vertical_range": float(pos[:, :, 1].max() - pos[:, :, 1].min()),
        "arm_motion_range": arm_range,
        "root_rotation_amount": float(np.abs(np.diff(yaw)).sum()),
    }
    return metrics


STYLE_CASES = {
    5: {
        "case": "slow_deliberate",
        "prompt": "A person walks forward slowly and deliberately.",
        "prefer_lower": [
            "root_speed_mean",
            "mean_joint_velocity",
            "mean_joint_acceleration",
            "mean_joint_jerk",
        ],
        "prefer_higher": [],
    },
    6: {
        "case": "graceful_dance",
        "prompt": "A person performs a graceful dance motion.",
        "prefer_lower": ["mean_joint_jerk"],
        "prefer_higher": [
            "arm_motion_range",
            "root_rotation_amount",
            "root_vertical_range",
        ],
    },
    7: {
        "case": "tai_chi",
        "prompt": "A person moves like doing tai chi.",
        "prefer_lower": ["root_speed_mean"],
        "prefer_higher": ["duration_sec", "arm_motion_range", "root_rotation_amount"],
    },
    8: {
        "case": "energetic_jump",
        "prompt": "A person jumps energetically.",
        "prefer_lower": [],
        "prefer_higher": [
            "root_vertical_range",
            "all_joint_vertical_range",
            "root_speed_mean",
            "arm_motion_range",
            "mean_joint_velocity",
            "mean_joint_acceleration",
        ],
    },
    9: {
        "case": "heavy_tired",
        "prompt": "A person walks in a tired and heavy style.",
        "prefer_lower": [
            "root_speed_mean",
            "mean_joint_velocity",
            "root_height_mean",
            "mean_joint_jerk",
        ],
        "prefer_higher": [],
    },
}


def score_case(baseline: dict[str, float | str], lora: dict[str, float | str], spec: dict) -> dict[str, float | str]:
    wins = 0
    total = 0
    details = {}
    for key in spec["prefer_lower"]:
        b = float(baseline[key])
        l = float(lora[key])
        delta_pct = (b - l) / max(abs(b), 1e-8) * 100.0
        details[f"{key}_delta_pct"] = delta_pct
        wins += int(l < b)
        total += 1
    for key in spec["prefer_higher"]:
        b = float(baseline[key])
        l = float(lora[key])
        delta_pct = (l - b) / max(abs(b), 1e-8) * 100.0
        details[f"{key}_delta_pct"] = delta_pct
        wins += int(l > b)
        total += 1
    return {
        "case": spec["case"],
        "prompt": spec["prompt"],
        "style_proxy_wins": wins,
        "style_proxy_total": total,
        "style_proxy_win_rate": wins / total if total else 0.0,
        **details,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline/project_prompts"))
    parser.add_argument("--lora-dir", type=Path, default=Path("outputs/finetune_lora_200_samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metrics_lora_200"))
    args = parser.parse_args()

    rows = []
    comparison_rows = []
    score_rows = []
    for idx in range(10):
        baseline_path = args.baseline_dir / f"evaluate_gpt{idx}.bvh"
        lora_path = args.lora_dir / f"evaluate_lora{idx}.bvh"
        if not baseline_path.exists() or not lora_path.exists():
            print(f"missing pair {idx}", file=sys.stderr)
            continue
        baseline = compute_metrics(baseline_path)
        lora = compute_metrics(lora_path)
        baseline.update({"model": "baseline", "prompt_index": idx})
        lora.update({"model": "lora", "prompt_index": idx})
        rows.extend([baseline, lora])
        compare = {"prompt_index": idx}
        for key in baseline:
            if key in {"file", "model", "prompt_index"}:
                continue
            b = baseline[key]
            l = lora[key]
            if isinstance(b, float) and isinstance(l, float):
                compare[f"{key}_baseline"] = b
                compare[f"{key}_lora"] = l
                compare[f"{key}_lora_minus_baseline"] = l - b
        comparison_rows.append(compare)
        if idx in STYLE_CASES:
            score_rows.append(score_case(baseline, lora, STYLE_CASES[idx]))

    write_csv(args.output_dir / "bvh_proxy_metrics.csv", rows)
    write_csv(args.output_dir / "bvh_proxy_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "style_proxy_scores.csv", score_rows)
    if score_rows:
        mean_win = sum(float(row["style_proxy_win_rate"]) for row in score_rows) / len(score_rows)
        print(f"style_proxy_cases={len(score_rows)} mean_win_rate={mean_win:.3f}")
    print(f"wrote metrics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
