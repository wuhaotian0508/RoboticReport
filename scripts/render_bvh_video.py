"""Render a BVH file to a simple stick-figure MP4/GIF without Blender."""

from __future__ import annotations

import argparse
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


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
    edges: list[tuple[str, str]] = []
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

        offset = np.zeros(3, dtype=np.float64)
        channels: list[str] = []
        children: list[Joint] = []
        joint = Joint(name=name, offset=offset, channels=channels, children=children, parent=parent)
        all_joints.append(joint)
        if parent is not None:
            edges.append((parent.name, joint.name))

        while idx < len(lines):
            line = lines[idx]
            if line.startswith("OFFSET"):
                offset_vals = [float(x) for x in line.split()[1:4]]
                joint.offset = np.array(offset_vals, dtype=np.float64)
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
    return root, channel_joints, all_joints, edges, motion, frame_time


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


def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # BVH uses Y-up. Matplotlib plot uses Z-up for a more natural view.
    return points[:, 0], points[:, 2], points[:, 1]


def render_frame(positions, edges, output_path: Path, bounds, title: str, dpi: int):
    fig = plt.figure(figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for parent, child in edges:
        pts = np.stack([positions[parent], positions[child]], axis=0)
        xs, ys, zs = project(pts)
        ax.plot(xs, ys, zs, color="#1f2937", linewidth=2.2)

    pts = np.stack(list(positions.values()), axis=0)
    xs, ys, zs = project(pts)
    ax.scatter(xs, ys, zs, color="#ef4444", s=8)

    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])
    ax.view_init(elev=18, azim=-62)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def compute_bounds(root, motion, sample_indices):
    all_points = []
    for idx in sample_indices:
        pose = frame_pose(root, motion[idx])
        all_points.append(np.stack(list(pose.values()), axis=0))
    pts = np.concatenate(all_points, axis=0)
    xs, ys, zs = project(pts)
    center = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, (zs.min() + zs.max()) / 2])
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min(), 1.0)
    radius = span * 0.58
    return {
        "x": (center[0] - radius, center[0] + radius),
        "y": (center[1] - radius, center[1] + radius),
        "z": (max(0.0, center[2] - radius), center[2] + radius),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bvh", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/rendered_motion.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    root, _, _, edges, motion, frame_time = parse_bvh(args.bvh)
    source_fps = 1.0 / frame_time
    step = max(1, round(source_fps / args.fps))
    indices = list(range(0, len(motion), step))[: args.max_frames]
    if not indices:
        raise RuntimeError("No frames to render.")

    bounds = compute_bounds(root, motion, indices[:: max(1, len(indices) // 30)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or args.bvh.name

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frame_paths = []
        for out_idx, frame_idx in enumerate(indices):
            pose = frame_pose(root, motion[frame_idx])
            png = tmp_path / f"frame_{out_idx:05d}.png"
            render_frame(pose, edges, png, bounds, f"{title} | frame {frame_idx}", args.dpi)
            frame_paths.append(png)
            if (out_idx + 1) % 25 == 0 or out_idx + 1 == len(indices):
                print(f"rendered {out_idx + 1}/{len(indices)} frames")

        images = [imageio.imread(path) for path in frame_paths]
        if args.output.suffix.lower() == ".gif":
            imageio.mimsave(args.output, images, fps=args.fps)
        else:
            try:
                imageio.mimsave(args.output, images, fps=args.fps, codec="libx264", quality=8)
            except Exception as exc:  # noqa: BLE001
                fallback = args.output.with_suffix(".gif")
                print(f"mp4 encode failed ({exc}); writing {fallback}")
                imageio.mimsave(fallback, images, fps=args.fps)
                args.output = fallback

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
