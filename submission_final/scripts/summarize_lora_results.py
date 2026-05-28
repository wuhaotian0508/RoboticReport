"""Summarize LoRA distillation outputs for reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_losses(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_epoch_summary(rows: list[dict[str, str]], output: Path) -> None:
    by_epoch: dict[int, list[float]] = {}
    for row in rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(float(row["loss"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "steps", "mean_loss", "min_loss", "max_loss", "last_loss"],
        )
        writer.writeheader()
        for epoch, losses in sorted(by_epoch.items()):
            writer.writerow(
                {
                    "epoch": epoch,
                    "steps": len(losses),
                    "mean_loss": f"{sum(losses) / len(losses):.6f}",
                    "min_loss": f"{min(losses):.6f}",
                    "max_loss": f"{max(losses):.6f}",
                    "last_loss": f"{losses[-1]:.6f}",
                }
            )


def plot_loss(rows: list[dict[str, str]], output: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.plot(steps, losses, color="#2563eb", linewidth=1.4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("LoRA style distillation training loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_train_val_loss(path: Path, output: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    val = [float(row["val_loss"]) for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(epochs, train, marker="o", label="Train cache")
    ax.plot(epochs, val, marker="o", label="Validation cache")
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Mean distillation loss")
    ax.set_title("LoRA train/validation distillation loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def video_frame(path: Path, fraction: float = 0.45):
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        nframes = meta.get("nframes")
        if not nframes or nframes == float("inf"):
            nframes = reader.count_frames()
        idx = max(0, min(int(nframes * fraction), nframes - 1))
        return reader.get_data(idx)
    finally:
        reader.close()


def make_qualitative_grid(pairs: list[tuple[str, Path, Path]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 4 * len(pairs)), dpi=150)
    if len(pairs) == 1:
        axes = [axes]

    for row_idx, (label, baseline, lora) in enumerate(pairs):
        for col_idx, (title, path) in enumerate((("Baseline", baseline), ("LoRA adapter", lora))):
            frame = video_frame(path)
            ax = axes[row_idx][col_idx]
            ax.imshow(frame)
            ax.set_title(f"{label} - {title}", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_manifest(pairs: list[tuple[str, Path, Path]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "baseline_video", "lora_video"])
        writer.writeheader()
        for label, baseline, lora in pairs:
            writer.writerow(
                {
                    "case": label,
                    "baseline_video": str(baseline),
                    "lora_video": str(lora),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/finetune_lora_200"))
    parser.add_argument("--comparison-dir", type=Path, default=Path("outputs/comparison_lora_200"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()

    rows = read_losses(args.run_dir / "train_log.csv")
    write_epoch_summary(rows, args.tables_dir / "lora_training_summary.csv")
    plot_loss(rows, args.figures_dir / "lora_loss_curve.png")
    plot_train_val_loss(
        args.tables_dir / "lora_train_val_loss.csv",
        args.figures_dir / "lora_train_val_loss.png",
    )

    if (args.comparison_dir / "baseline_dance.mp4").exists():
        pairs = [
            (
                "Slow deliberate walk",
                args.comparison_dir / "baseline_slow.mp4",
                args.comparison_dir / "lora_slow.mp4",
            ),
            (
                "Graceful dance",
                args.comparison_dir / "baseline_dance.mp4",
                args.comparison_dir / "lora_dance.mp4",
            ),
            (
                "Energetic jump",
                args.comparison_dir / "baseline_jump.mp4",
                args.comparison_dir / "lora_jump.mp4",
            ),
        ]
    else:
        pairs = [
            (
                "Slow deliberate walk",
                args.comparison_dir / "baseline_slow.mp4",
                args.comparison_dir / "lora_slow.mp4",
            ),
            (
                "Tai chi style",
                args.comparison_dir / "baseline_taichi.mp4",
                args.comparison_dir / "lora_taichi.mp4",
            ),
            (
                "Heavy tired walk",
                args.comparison_dir / "baseline_heavy.mp4",
                args.comparison_dir / "lora_heavy.mp4",
            ),
        ]
    write_manifest(pairs, args.comparison_dir / "comparison_manifest.csv")
    make_qualitative_grid(pairs, args.figures_dir / "lora_qualitative_frames.png")

    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    print(
        "summarized LoRA run: "
        f"examples={config['examples']} trainable={config['trainable_params']} "
        f"fraction={config['trainable_fraction']:.4%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
