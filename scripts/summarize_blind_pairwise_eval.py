"""Summarize blind baseline-vs-LoRA pairwise human evaluation labels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = [
    "semantic_a_1_to_5",
    "semantic_b_1_to_5",
    "style_a_1_to_5",
    "style_b_1_to_5",
    "plausibility_a_1_to_5",
    "plausibility_b_1_to_5",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(key_rows: list[dict[str, str]], label_rows: list[dict[str, str]]):
    key_by_item = {row["item_id"]: row for row in key_rows}
    per_item = []
    scores: dict[str, list[float]] = defaultdict(list)
    preference_counts = {"baseline": 0, "lora": 0, "tie": 0, "missing": 0}

    for label in label_rows:
        item_id = (label.get("item_id") or "").strip()
        if not item_id:
            continue
        key = key_by_item.get(item_id)
        if key is None:
            raise RuntimeError(f"Label references unknown item_id: {item_id}")

        pref = (label.get("preference_a_b_tie") or "").strip().lower()
        if pref in {"a", "left"}:
            winner = key["a_model"]
        elif pref in {"b", "right"}:
            winner = key["b_model"]
        elif pref == "tie":
            winner = "tie"
        else:
            winner = "missing"
        preference_counts[winner] += 1

        mapped = {
            "item_id": item_id,
            "prompt_index": key["prompt_index"],
            "prompt": key["prompt"],
            "evaluator": label.get("evaluator", ""),
            "a_model": key["a_model"],
            "b_model": key["b_model"],
            "preference_a_b_tie": label.get("preference_a_b_tie", ""),
            "winner_model": winner,
            "notes": label.get("notes", ""),
        }
        for metric in ("semantic", "style", "plausibility"):
            for side, model in (("a", key["a_model"]), ("b", key["b_model"])):
                value = as_float(label.get(f"{metric}_{side}_1_to_5", ""))
                if value is not None:
                    scores[f"{model}_{metric}"].append(value)
                    mapped[f"{model}_{metric}_score"] = value
        per_item.append(mapped)

    completed = preference_counts["baseline"] + preference_counts["lora"] + preference_counts["tie"]
    non_ties = preference_counts["baseline"] + preference_counts["lora"]
    summary = {
        "num_labels": len(label_rows),
        "num_completed_preferences": completed,
        "baseline_wins": preference_counts["baseline"],
        "lora_wins": preference_counts["lora"],
        "ties": preference_counts["tie"],
        "missing_preferences": preference_counts["missing"],
        "lora_win_rate_excluding_ties": preference_counts["lora"] / non_ties if non_ties else 0.0,
        "lora_preference_rate_including_ties_half": (
            (preference_counts["lora"] + 0.5 * preference_counts["tie"]) / completed
            if completed
            else 0.0
        ),
        "baseline_semantic_mean": mean(scores["baseline_semantic"]),
        "lora_semantic_mean": mean(scores["lora_semantic"]),
        "baseline_style_mean": mean(scores["baseline_style"]),
        "lora_style_mean": mean(scores["lora_style"]),
        "baseline_plausibility_mean": mean(scores["baseline_plausibility"]),
        "lora_plausibility_mean": mean(scores["lora_plausibility"]),
    }
    return per_item, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-key", type=Path, default=Path("outputs/human_pairwise_eval/blind_key.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/human_pairwise_eval/preference_labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/human_pairwise_eval"))
    args = parser.parse_args()

    per_item, summary = summarize(read_csv(args.blind_key), read_csv(args.labels))
    write_csv(args.output_dir / "per_item_results.csv", per_item)
    write_csv(args.output_dir / "summary.csv", [summary])
    print(f"wrote {args.output_dir / 'per_item_results.csv'}")
    print(f"wrote {args.output_dir / 'summary.csv'}")
    print(
        "LoRA preference rate including ties as half wins: "
        f"{summary['lora_preference_rate_including_ties_half']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
