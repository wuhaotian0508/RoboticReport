"""Export captions from a CSV file to a plain prompt list."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-csv", type=Path, default=Path("data/style_subset_small/style_test_small.csv"))
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--output", type=Path, default=Path("outputs/human_pairwise_eval/test_prompts.txt"))
    parser.add_argument("--max-items", type=int, default=40)
    args = parser.parse_args()

    prompts: list[str] = []
    with args.prompt_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if args.caption_column not in (reader.fieldnames or []):
            raise RuntimeError(f"Missing caption column `{args.caption_column}` in {args.prompt_csv}")
        for row in reader:
            caption = (row.get(args.caption_column) or "").strip()
            if caption:
                prompts.append(caption)
            if args.max_items and len(prompts) >= args.max_items:
                break

    if not prompts:
        raise RuntimeError(f"No prompts exported from {args.prompt_csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(prompts)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
