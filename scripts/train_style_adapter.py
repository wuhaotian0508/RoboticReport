"""Minimal fine-tuning entry point with honest diagnostics.

The upstream MoConVQ release includes pretrained MoConGPT inference weights but
does not include the original MoConGPT training code. This script therefore
serves as the controlled midterm entry point: it validates the style subset and
token requirements, writes a diagnostic report, and only proceeds to training
when tokenized MoConVQ targets are available.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


REQUIRED_COLUMNS = {"caption", "motion_id"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moconvq-root", type=Path, default=Path("MoConVQ"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/style_subset_small/style_train_small.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/style_subset_small/style_val_small.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--token-column", default="token_file")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    train_fields, train_rows = read_csv(args.train_csv)
    val_fields, val_rows = read_csv(args.val_csv)
    fields = set(train_fields)
    missing_columns = sorted(REQUIRED_COLUMNS - fields)
    token_ready = bool(train_rows) and args.token_column in fields and all(
        row.get(args.token_column, "").strip() for row in train_rows
    )

    checkpoints = {
        "moconvq_base.data": args.moconvq_root / "moconvq_base.data",
        "text_generation_GPT.pth": args.moconvq_root / "text_generation_GPT.pth",
        "simple_motion_data.h5": args.moconvq_root / "simple_motion_data.h5",
    }
    missing_files = [name for name, path in checkpoints.items() if not path.exists()]

    lines = [
        "# Style Adapter Training Diagnostic",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        "",
        f"- MoConVQ root: `{args.moconvq_root}`",
        f"- Train CSV: `{args.train_csv}` ({len(train_rows)} rows)",
        f"- Val CSV: `{args.val_csv}` ({len(val_rows)} rows)",
        f"- Planned max steps: {args.max_steps}",
        f"- Batch size: {args.batch_size}",
        f"- Gradient accumulation: {args.grad_accum_steps}",
        f"- fp16 requested: {args.fp16}",
        "",
        "## Status",
        "",
    ]
    if missing_files:
        lines.append(f"- Missing MoConVQ files: {', '.join(missing_files)}")
    else:
        lines.append("- MoConVQ pretrained files are present.")
    if missing_columns:
        lines.append(f"- Training CSV is missing columns: {', '.join(missing_columns)}")
    elif not train_rows:
        lines.append("- Training CSV has no rows yet; HumanML3D style subset has not been populated.")
    else:
        lines.append("- Training CSV has the required caption/motion_id metadata.")
    if token_ready:
        lines.append("- Token targets are present; a real training loop can be enabled here.")
    else:
        lines.append(
            "- Token targets are not present. Fine-tuning is paused because MoConGPT needs ground-truth MoConVQ token sequences."
        )
    lines += [
        "",
        "## Midterm Interpretation",
        "",
        "Use this as a diagnostic result if tokenization or HumanML3D preparation is not complete by the midterm deadline. Do not report FID, R-Precision, or loss values until measured.",
    ]

    report_path = args.output_dir / "training_diagnostic.md"
    write_report(report_path, lines)
    print(f"Wrote {report_path}")

    ready = not missing_files and not missing_columns and token_ready
    if ready:
        print("Tokenized targets are available; implement/enable the optimization loop before final experiments.")
        return 0
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
