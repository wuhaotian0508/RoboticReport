"""Sample small style-subset CSVs for quick training and manual inspection."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


SPLIT_CONFIG = {
    "train": ("style_subset_train.csv", "style_train_small.csv", "train_size"),
    "val": ("style_subset_val.csv", "style_val_small.csv", "val_size"),
    "test": ("style_subset_test.csv", "style_test_small.csv", "test_size"),
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/style_subset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/style_subset_small"))
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--val-size", type=int, default=40)
    parser.add_argument("--test-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    wrote_any = False
    for split, (input_name, output_name, size_attr) in SPLIT_CONFIG.items():
        fieldnames, rows = read_rows(args.input_dir / input_name)
        if not fieldnames:
            fieldnames = [
                "split",
                "motion_id",
                "line_index",
                "caption",
                "tokens",
                "f_tag",
                "to_tag",
                "style_groups",
                "matched_keywords",
                "text_path",
            ]
        rng.shuffle(rows)
        size = getattr(args, size_attr)
        sampled = rows[: min(size, len(rows))]
        write_rows(args.output_dir / output_name, fieldnames, sampled)
        print(f"{split}: wrote {len(sampled)} rows to {args.output_dir / output_name}")
        wrote_any = wrote_any or bool(sampled)
    if not wrote_any:
        print("No rows were sampled. This is expected before HumanML3D style filtering is populated.")
    return 0 if wrote_any or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
