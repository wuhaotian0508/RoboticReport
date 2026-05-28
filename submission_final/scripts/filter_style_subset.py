"""Build style-focused HumanML3D subsets without crossing official splits.

Expected HumanML3D layout, official text format:

    D:/roboticsreport/datasets/HumanML3D/
      texts/
        000001.txt
        ...
      train.txt
      val.txt
      test.txt

It also supports the Hugging Face parquet export:

    D:/roboticsreport/HumanML3D/
      data/
        train-*.parquet
        val-*.parquet
        test-*.parquet

HumanML3D text rows are usually formatted as:
    caption#tokenized_caption#f_tag#to_tag

The script filters captions independently inside train/val/test, writes one CSV
per split, and creates a manual-audit sample for the report. If the dataset is
not present yet, use --allow-missing to create diagnostic placeholder files.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path(r"D:\roboticsreport\datasets\HumanML3D")
DEFAULT_KEYWORDS = Path("data/style_keywords.csv")
DEFAULT_OUTPUT = Path("data/style_subset")
DEFAULT_AUDIT = Path("data/manual_audit_sample.csv")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class StyleGroup:
    name: str
    keywords: tuple[str, ...]


def read_style_groups(path: Path) -> list[StyleGroup]:
    groups: list[StyleGroup] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("style_group") or "").strip()
            keywords = tuple(
                k.strip().lower()
                for k in (row.get("keywords") or "").split(";")
                if k.strip()
            )
            if name and keywords:
                groups.append(StyleGroup(name=name, keywords=keywords))
    if not groups:
        raise ValueError(f"No style keywords found in {path}")
    return groups


def keyword_matches(caption: str, keyword: str) -> bool:
    text = caption.lower()
    escaped = re.escape(keyword.lower())
    if " " in keyword:
        return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def read_split_ids(split_path: Path) -> list[str]:
    with split_path.open("r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def parse_text_file(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_index, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split("#")
            caption = parts[0].strip()
            tokens = parts[1].strip() if len(parts) > 1 else ""
            f_tag = parts[2].strip() if len(parts) > 2 else ""
            to_tag = parts[3].strip() if len(parts) > 3 else ""
            yield {
                "line_index": str(line_index),
                "caption": caption,
                "tokens": tokens,
                "f_tag": f_tag,
                "to_tag": to_tag,
            }


def parse_caption_blob(blob: str) -> Iterable[dict[str, str]]:
    for line_index, raw in enumerate((blob or "").splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("#")
        caption = parts[0].strip()
        tokens = parts[1].strip() if len(parts) > 1 else ""
        f_tag = parts[2].strip() if len(parts) > 2 else ""
        to_tag = parts[3].strip() if len(parts) > 3 else ""
        yield {
            "line_index": str(line_index),
            "caption": caption,
            "tokens": tokens,
            "f_tag": f_tag,
            "to_tag": to_tag,
        }


def match_caption(caption: str, groups: list[StyleGroup]) -> tuple[list[str], list[str]]:
    matched_groups: list[str] = []
    matched_keywords: list[str] = []
    for group in groups:
        group_hits = [kw for kw in group.keywords if keyword_matches(caption, kw)]
        if group_hits:
            matched_groups.append(group.name)
            matched_keywords.extend(group_hits)
    return matched_groups, sorted(set(matched_keywords))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_missing_diagnostic(
    output_dir: Path,
    audit_path: Path,
    root: Path,
    missing: list[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    for split in SPLITS:
        write_csv(output_dir / f"style_subset_{split}.csv", [], fieldnames)
    write_csv(
        output_dir / "style_subset_statistics.csv",
        [
            {
                "split": "ALL",
                "style_group": "DATASET_MISSING",
                "matched_caption_rows": "0",
                "matched_motion_ids": "0",
                "total_motion_ids": "0",
                "total_caption_rows": "0",
            }
        ],
        [
            "split",
            "style_group",
            "matched_caption_rows",
            "matched_motion_ids",
            "total_motion_ids",
            "total_caption_rows",
        ],
    )
    write_csv(
        audit_path,
        [],
        [
            "split",
            "motion_id",
            "caption",
            "style_groups",
            "matched_keywords",
            "audit_label",
            "notes",
        ],
    )
    missing_text = "\n".join(f"- {p}" for p in missing)
    (output_dir / "HUMANML3D_MISSING.md").write_text(
        "\n".join(
            [
                "# HumanML3D Missing",
                "",
                f"Expected root: `{root}`",
                "",
                "Missing required paths:",
                missing_text,
                "",
                "Place the official HumanML3D `texts/`, `train.txt`, `val.txt`, and `test.txt` files here, then rerun:",
                "",
                "```bash",
                "cd /d/roboticsreport",
                "./MoConVQ/.venv/Scripts/python.exe scripts/filter_style_subset.py",
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def collect_rows(root: Path, groups: list[StyleGroup]) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    rows_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    stats: list[dict[str, str]] = []
    texts_dir = root / "texts"

    for split in SPLITS:
        split_ids = read_split_ids(root / f"{split}.txt")
        total_caption_rows = 0
        matched_ids: set[str] = set()
        per_group_rows = {group.name: 0 for group in groups}
        per_group_ids = {group.name: set() for group in groups}

        for motion_id in split_ids:
            text_path = texts_dir / f"{motion_id}.txt"
            if not text_path.exists():
                continue
            for item in parse_text_file(text_path):
                total_caption_rows += 1
                matched_groups, matched_keywords = match_caption(item["caption"], groups)
                if not matched_groups:
                    continue
                matched_ids.add(motion_id)
                for group in matched_groups:
                    per_group_rows[group] += 1
                    per_group_ids[group].add(motion_id)
                rows_by_split[split].append(
                    {
                        "split": split,
                        "motion_id": motion_id,
                        "line_index": item["line_index"],
                        "caption": item["caption"],
                        "tokens": item["tokens"],
                        "f_tag": item["f_tag"],
                        "to_tag": item["to_tag"],
                        "style_groups": "|".join(matched_groups),
                        "matched_keywords": ";".join(matched_keywords),
                        "text_path": str(text_path),
                    }
                )

        stats.append(
            {
                "split": split,
                "style_group": "ALL",
                "matched_caption_rows": str(len(rows_by_split[split])),
                "matched_motion_ids": str(len(matched_ids)),
                "total_motion_ids": str(len(split_ids)),
                "total_caption_rows": str(total_caption_rows),
            }
        )
        for group in groups:
            stats.append(
                {
                    "split": split,
                    "style_group": group.name,
                    "matched_caption_rows": str(per_group_rows[group.name]),
                    "matched_motion_ids": str(len(per_group_ids[group.name])),
                    "total_motion_ids": str(len(split_ids)),
                    "total_caption_rows": str(total_caption_rows),
                }
            )
    return rows_by_split, stats


def parquet_files_by_split(root: Path) -> dict[str, list[Path]]:
    data_dir = root / "data"
    return {split: sorted(data_dir.glob(f"{split}-*.parquet")) for split in SPLITS}


def collect_rows_from_parquet(root: Path, groups: list[StyleGroup]) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet HumanML3D. Install it with: "
            "uv pip install --python ./MoConVQ/.venv/Scripts/python.exe pyarrow"
        ) from exc

    rows_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    stats: list[dict[str, str]] = []
    files_by_split = parquet_files_by_split(root)

    for split in SPLITS:
        total_caption_rows = 0
        all_motion_ids: set[str] = set()
        matched_ids: set[str] = set()
        per_group_rows = {group.name: 0 for group in groups}
        per_group_ids = {group.name: set() for group in groups}

        for parquet_path in files_by_split[split]:
            parquet_file = pq.ParquetFile(parquet_path)
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_index, columns=["caption", "meta_data"])
                for row_index, item in enumerate(table.to_pylist()):
                    meta = item.get("meta_data") or {}
                    motion_id = str(meta.get("name") or f"{parquet_path.stem}_{row_group_index}_{row_index}")
                    all_motion_ids.add(motion_id)
                    for parsed in parse_caption_blob(item.get("caption") or ""):
                        total_caption_rows += 1
                        matched_groups, matched_keywords = match_caption(parsed["caption"], groups)
                        if not matched_groups:
                            continue
                        matched_ids.add(motion_id)
                        for group in matched_groups:
                            per_group_rows[group] += 1
                            per_group_ids[group].add(motion_id)
                        rows_by_split[split].append(
                            {
                                "split": split,
                                "motion_id": motion_id,
                                "line_index": parsed["line_index"],
                                "caption": parsed["caption"],
                                "tokens": parsed["tokens"],
                                "f_tag": parsed["f_tag"],
                                "to_tag": parsed["to_tag"],
                                "style_groups": "|".join(matched_groups),
                                "matched_keywords": ";".join(matched_keywords),
                                "text_path": str(parquet_path),
                            }
                        )

        stats.append(
            {
                "split": split,
                "style_group": "ALL",
                "matched_caption_rows": str(len(rows_by_split[split])),
                "matched_motion_ids": str(len(matched_ids)),
                "total_motion_ids": str(len(all_motion_ids)),
                "total_caption_rows": str(total_caption_rows),
            }
        )
        for group in groups:
            stats.append(
                {
                    "split": split,
                    "style_group": group.name,
                    "matched_caption_rows": str(per_group_rows[group.name]),
                    "matched_motion_ids": str(len(per_group_ids[group.name])),
                    "total_motion_ids": str(len(all_motion_ids)),
                    "total_caption_rows": str(total_caption_rows),
                }
            )
    return rows_by_split, stats


def write_audit_sample(path: Path, rows: list[dict[str, str]], size: int, seed: int) -> None:
    rng = random.Random(seed)
    sample = list(rows)
    rng.shuffle(sample)
    sample = sample[: min(size, len(sample))]
    audit_rows = [
        {
            "split": row["split"],
            "motion_id": row["motion_id"],
            "caption": row["caption"],
            "style_groups": row["style_groups"],
            "matched_keywords": row["matched_keywords"],
            "audit_label": "",
            "notes": "",
        }
        for row in sample
    ]
    write_csv(
        path,
        audit_rows,
        [
            "split",
            "motion_id",
            "caption",
            "style_groups",
            "matched_keywords",
            "audit_label",
            "notes",
        ],
    )


def validate_required(root: Path, keyword_path: Path) -> list[Path]:
    if (root / "data").exists() and any((root / "data").glob("*.parquet")):
        missing = [keyword_path] if not keyword_path.exists() else []
        files_by_split = parquet_files_by_split(root)
        for split in SPLITS:
            if not files_by_split[split]:
                missing.append(root / "data" / f"{split}-*.parquet")
        return missing
    required = [keyword_path, root / "texts"] + [root / f"{split}.txt" for split in SPLITS]
    return [path for path in required if not path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--keyword-csv", type=Path, default=DEFAULT_KEYWORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--audit-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    missing = validate_required(args.humanml3d_root, args.keyword_csv)
    if missing:
        if args.allow_missing:
            write_missing_diagnostic(args.output_dir, args.audit_output, args.humanml3d_root, missing)
            print("HumanML3D is missing; wrote diagnostic placeholder files.")
            for path in missing:
                print(f"missing: {path}")
            return 0
        for path in missing:
            print(f"missing: {path}")
        return 2

    groups = read_style_groups(args.keyword_csv)
    if (args.humanml3d_root / "data").exists() and any((args.humanml3d_root / "data").glob("*.parquet")):
        rows_by_split, stats = collect_rows_from_parquet(args.humanml3d_root, groups)
    else:
        rows_by_split, stats = collect_rows(args.humanml3d_root, groups)
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
    for split, rows in rows_by_split.items():
        write_csv(args.output_dir / f"style_subset_{split}.csv", rows, fieldnames)
    write_csv(
        args.output_dir / "style_subset_statistics.csv",
        stats,
        [
            "split",
            "style_group",
            "matched_caption_rows",
            "matched_motion_ids",
            "total_motion_ids",
            "total_caption_rows",
        ],
    )
    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    write_audit_sample(args.audit_output, all_rows, args.audit_size, args.seed)
    print(f"Wrote style subsets to {args.output_dir}")
    print(f"Wrote audit sample to {args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
