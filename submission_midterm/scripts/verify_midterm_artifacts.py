"""Verify local baseline and midterm evidence without running generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(sum(1 for _ in csv.DictReader(f)), 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moconvq-root", type=Path, default=Path("MoConVQ"))
    parser.add_argument("--style-dir", type=Path, default=Path("data/style_subset"))
    parser.add_argument("--baseline-copy-dir", type=Path, default=Path("outputs/baseline/project_prompts"))
    parser.add_argument("--expected-prompts", type=int, default=10)
    args = parser.parse_args()

    manifest = args.moconvq_root / "out/project_prompts/baseline_manifest.csv"
    bvh_dir = args.moconvq_root / "out/project_prompts"
    baseline_rows = count_csv_rows(manifest)
    bvh_files = sorted(bvh_dir.glob("evaluate_gpt*.bvh")) if bvh_dir.exists() else []
    copied_bvh_files = (
        sorted(args.baseline_copy_dir.glob("evaluate_gpt*.bvh"))
        if args.baseline_copy_dir.exists()
        else []
    )

    checks = [
        ("moconvq_base.data", (args.moconvq_root / "moconvq_base.data").exists()),
        ("text_generation_GPT.pth", (args.moconvq_root / "text_generation_GPT.pth").exists()),
        ("simple_motion_data.h5", (args.moconvq_root / "simple_motion_data.h5").exists()),
        ("baseline manifest", baseline_rows >= args.expected_prompts),
        ("baseline BVH count", len(bvh_files) >= args.expected_prompts),
        ("baseline evidence copy", len(copied_bvh_files) >= args.expected_prompts),
    ]

    for split in ("train", "val", "test"):
        path = args.style_dir / f"style_subset_{split}.csv"
        rows = count_csv_rows(path)
        checks.append((f"style subset {split} exists", rows >= 0))
    stats_rows = count_csv_rows(args.style_dir / "style_subset_statistics.csv")
    checks.append(("style statistics exists", stats_rows >= 0))

    ok = True
    for name, passed in checks:
        print(f"[{'OK' if passed else 'MISS'}] {name}")
        ok = ok and passed
    print(f"baseline_manifest_rows={baseline_rows}")
    print(f"baseline_bvh_files={len(bvh_files)}")
    print(f"copied_baseline_bvh_files={len(copied_bvh_files)}")
    print(f"style_statistics_rows={stats_rows}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
