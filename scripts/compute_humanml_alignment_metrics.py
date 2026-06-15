"""Evaluate generated BVH motions against paired HumanML3D motions.

This complements style_proxy_win_rate.  The style proxy asks whether LoRA moved
metrics in a hand-written style direction relative to baseline; this script asks
whether a generated motion is closer to the HumanML3D motion paired with the
same caption row.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from compute_bvh_proxy_metrics import compute_metrics  # noqa: E402
from humanml_motion_proxy import (  # noqa: E402
    DEFAULT_SELECTION_METRICS,
    HumanMLMotionStore,
    metric_distance,
    read_style_rows,
)


def alignment_distance(
    generated_metrics: dict[str, float | str],
    target_metrics: dict[str, float | str],
    metric_names: tuple[str, ...] = DEFAULT_SELECTION_METRICS,
) -> float:
    distances = [
        metric_distance(generated_metrics, target_metrics, key)
        for key in metric_names
        if key in generated_metrics and key in target_metrics
    ]
    if not distances:
        return 0.0
    return float(sum(distances) / len(distances))


def alignment_score(distance: float) -> float:
    return float(math.exp(-max(float(distance), 0.0)))


def compare_alignment(
    row_index: int,
    caption: str,
    motion_id: str,
    target_metrics: dict[str, float | str],
    generated_metrics: dict[str, dict[str, float | str]],
    metric_names: tuple[str, ...] = DEFAULT_SELECTION_METRICS,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_index": row_index,
        "caption": caption,
        "motion_id": motion_id,
    }
    for key in metric_names:
        if key in target_metrics:
            row[f"humanml_{key}"] = float(target_metrics[key])

    distances: dict[str, float] = {}
    for label, metrics in generated_metrics.items():
        distance = alignment_distance(metrics, target_metrics, metric_names)
        distances[label] = distance
        row[f"{label}_alignment_distance"] = distance
        row[f"{label}_alignment_score"] = alignment_score(distance)
        for key in metric_names:
            if key in metrics and key in target_metrics:
                row[f"{label}_{key}_log_ratio_distance"] = metric_distance(metrics, target_metrics, key)

    if "baseline" in distances and "lora" in distances:
        row["lora_closer_to_humanml"] = int(distances["lora"] < distances["baseline"])
        row["alignment_distance_delta"] = distances["baseline"] - distances["lora"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generated_bvh_path(directory: Path, prefix: str, index: int) -> Path:
    return directory / f"{prefix}{index}.bvh"


def selected_candidate_bvh_path(candidate_dir: Path, row_number: int, selected_idx: int) -> Path:
    return candidate_dir / f"candidate_{row_number:05d}_{selected_idx:02d}.bvh"


def load_cache_examples(cache_path: Path | None) -> list[dict[str, Any]] | None:
    if cache_path is None:
        return None
    try:
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(cache_path, map_location="cpu")


def build_alignment_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    metric_names = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())
    rows = read_style_rows(args.style_csv, max_samples=args.max_samples, seed=args.seed)
    cache_examples = load_cache_examples(args.cache_path)
    if cache_examples is not None:
        rows = rows[: len(cache_examples)]

    store = HumanMLMotionStore()
    output_rows: list[dict[str, Any]] = []
    for zero_idx, style_row in enumerate(rows):
        motion_id = style_row.get("motion_id", "")
        text_path = style_row.get("text_path", "")
        if not motion_id or not text_path:
            continue
        try:
            target_metrics = store.metrics_for(Path(text_path), motion_id, fps=args.humanml_fps)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: skipping {motion_id}: {exc}")
            continue

        generated: dict[str, dict[str, float | str]] = {}
        if args.baseline_dir:
            path = generated_bvh_path(args.baseline_dir, args.baseline_prefix, zero_idx)
            if path.exists():
                generated["baseline"] = compute_metrics(path)
        if args.lora_dir:
            path = generated_bvh_path(args.lora_dir, args.lora_prefix, zero_idx)
            if path.exists():
                generated["lora"] = compute_metrics(path)
        if args.candidate_bvh_dir and cache_examples is not None:
            selected_idx = int(cache_examples[zero_idx].get("selected_sample_idx", 0))
            path = selected_candidate_bvh_path(args.candidate_bvh_dir, zero_idx + 1, selected_idx)
            if path.exists():
                generated["selected_teacher"] = compute_metrics(path)

        if not generated:
            continue
        output_rows.append(
            compare_alignment(
                row_index=zero_idx,
                caption=style_row.get("caption", ""),
                motion_id=motion_id,
                target_metrics=target_metrics,
                generated_metrics=generated,
                metric_names=metric_names,
            )
        )
    return output_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": len(rows)}
    labels = sorted(
        {
            key[: -len("_alignment_distance")]
            for row in rows
            for key in row
            if key.endswith("_alignment_distance")
        }
    )
    for label in labels:
        distances = [float(row[f"{label}_alignment_distance"]) for row in rows if f"{label}_alignment_distance" in row]
        scores = [float(row[f"{label}_alignment_score"]) for row in rows if f"{label}_alignment_score" in row]
        if distances:
            summary[f"{label}_mean_alignment_distance"] = sum(distances) / len(distances)
            summary[f"{label}_mean_alignment_score"] = sum(scores) / len(scores)
    wins = [int(row["lora_closer_to_humanml"]) for row in rows if "lora_closer_to_humanml" in row]
    if wins:
        summary["lora_humanml_win_rate"] = sum(wins) / len(wins)
        summary["lora_humanml_wins"] = sum(wins)
        summary["lora_humanml_total"] = len(wins)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-csv", type=Path, default=Path("data/style_subset_small/style_train_small.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/humanml_alignment"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--humanml-fps", type=float, default=20.0)
    parser.add_argument("--metrics", default=",".join(DEFAULT_SELECTION_METRICS))
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--baseline-prefix", default="evaluate_gpt")
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument("--lora-prefix", default="evaluate_lora")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--candidate-bvh-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = build_alignment_rows(args)
    if not rows:
        raise RuntimeError("No aligned generated motions were found.")
    summary = summarize(rows)

    write_csv(args.output_dir / "humanml_alignment_metrics.csv", rows)
    write_csv(args.output_dir / "humanml_alignment_summary.csv", [summary])
    print(f"rows={summary['rows']}")
    for key, value in summary.items():
        if key != "rows":
            print(f"{key}={value}")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
