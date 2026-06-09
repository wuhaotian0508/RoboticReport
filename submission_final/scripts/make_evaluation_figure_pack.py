"""Create paper-ready evaluation figures from existing LoRA metric CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SUMMARY_FIELDS = [
    "style_proxy_cases",
    "total_style_proxy_wins",
    "total_style_proxy_checks",
    "overall_style_proxy_win_rate",
    "mean_case_style_proxy_win_rate",
    "best_case",
    "best_case_win_rate",
    "weakest_case",
    "weakest_case_win_rate",
    "main_final_train_loss",
    "main_final_val_loss",
    "continued_final_train_loss",
    "continued_final_val_loss",
]


METRIC_LABELS = {
    "all_joint_vertical_range_delta_pct": "All-joint\nvertical",
    "arm_motion_range_delta_pct": "Arm\nrange",
    "duration_sec_delta_pct": "Duration",
    "mean_joint_acceleration_delta_pct": "Joint\naccel.",
    "mean_joint_jerk_delta_pct": "Joint\njerk",
    "mean_joint_velocity_delta_pct": "Joint\nvelocity",
    "root_height_mean_delta_pct": "Root\nheight",
    "root_rotation_amount_delta_pct": "Root\nrotation",
    "root_speed_mean_delta_pct": "Root\nspeed",
    "root_vertical_range_delta_pct": "Root\nvertical",
}


def read_csv_rows(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required input missing: {path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_int(value: str | None) -> int:
    number = as_float(value)
    return int(number) if number is not None else 0


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.6f}"


def final_loss_values(path: Path) -> tuple[float | None, float | None]:
    rows = read_csv_rows(path, required=False)
    if not rows:
        return None, None

    def epoch_key(row: dict[str, str]) -> int:
        return as_int(row.get("epoch"))

    final = max(rows, key=epoch_key)
    return as_float(final.get("train_loss")), as_float(final.get("val_loss"))


def write_summary(path: Path, summary: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(summary)


def plot_scorecard(rows: list[dict[str, str]], output: Path) -> None:
    cases = [row["case"] for row in rows]
    rates = [(as_float(row.get("style_proxy_win_rate")) or 0.0) * 100.0 for row in rows]
    wins = [as_int(row.get("style_proxy_wins")) for row in rows]
    totals = [as_int(row.get("style_proxy_total")) for row in rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=180)
    y_positions = list(range(len(cases)))
    colors = ["#2f80ed" if rate < 90 else "#219653" for rate in rates]
    ax.barh(y_positions, rates, color=colors, height=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([case.replace("_", " ") for case in cases])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Expected-direction proxy win rate (%)")
    ax.set_title("LoRA style proxy scorecard")
    ax.grid(axis="x", alpha=0.2)
    for y, rate, win, total in zip(y_positions, rates, wins, totals):
        ax.text(min(rate + 2, 101), y, f"{win}/{total}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def delta_columns(rows: list[dict[str, str]]) -> list[str]:
    seen = []
    for key in METRIC_LABELS:
        if any((row.get(key) or "").strip() for row in rows):
            seen.append(key)
    return seen


def plot_delta_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    columns = delta_columns(rows)
    cases = [row["case"].replace("_", " ") for row in rows]
    data: list[list[float]] = []
    max_abs = 1.0
    for row in rows:
        line = []
        for column in columns:
            value = as_float(row.get(column))
            if value is None:
                line.append(float("nan"))
            else:
                line.append(value)
                max_abs = max(max_abs, abs(value))
        data.append(line)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(7.5, 0.82 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_width, 3.9), dpi=180)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#f1f5f9")
    image = ax.imshow(data, cmap=cmap, vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([METRIC_LABELS[column] for column in columns], fontsize=8)
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels(cases)
    ax.set_title("Expected-direction proxy metric changes")
    for row_idx, row in enumerate(data):
        for col_idx, value in enumerate(row):
            if math.isnan(value):
                continue
            color = "white" if abs(value) > max_abs * 0.55 else "#111827"
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Desired-direction change (%)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def read_loss_series(path: Path) -> list[dict[str, float]]:
    rows = read_csv_rows(path, required=False)
    series = []
    for row in rows:
        epoch = as_float(row.get("epoch"))
        train = as_float(row.get("train_loss"))
        val = as_float(row.get("val_loss"))
        if epoch is None or train is None or val is None:
            continue
        series.append({"epoch": epoch, "train_loss": train, "val_loss": val})
    return sorted(series, key=lambda row: row["epoch"])


def plot_distill_loss(main_path: Path, continued_path: Path, output: Path) -> bool:
    main = read_loss_series(main_path)
    continued = read_loss_series(continued_path)
    if not main and not continued:
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    if main:
        ax.plot(
            [row["epoch"] for row in main],
            [row["train_loss"] for row in main],
            marker="o",
            color="#2f80ed",
            label="Main train",
        )
        ax.plot(
            [row["epoch"] for row in main],
            [row["val_loss"] for row in main],
            marker="o",
            linestyle="--",
            color="#56ccf2",
            label="Main val",
        )
    if continued:
        ax.plot(
            [row["epoch"] for row in continued],
            [row["train_loss"] for row in continued],
            marker="s",
            color="#219653",
            label="Continued train",
        )
        ax.plot(
            [row["epoch"] for row in continued],
            [row["val_loss"] for row in continued],
            marker="s",
            linestyle="--",
            color="#6fcf97",
            label="Continued val",
        )
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Mean distillation loss")
    ax.set_title("Train/validation pseudo-token loss")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return True


def build_summary(rows: list[dict[str, str]], tables_dir: Path) -> dict[str, str]:
    total_wins = sum(as_int(row.get("style_proxy_wins")) for row in rows)
    total_checks = sum(as_int(row.get("style_proxy_total")) for row in rows)
    rates = [as_float(row.get("style_proxy_win_rate")) or 0.0 for row in rows]
    best = max(rows, key=lambda row: as_float(row.get("style_proxy_win_rate")) or 0.0)
    weakest = min(rows, key=lambda row: as_float(row.get("style_proxy_win_rate")) or 0.0)
    main_train, main_val = final_loss_values(tables_dir / "lora_train_val_loss.csv")
    cont_train, cont_val = final_loss_values(tables_dir / "lora_cont_train_val_loss.csv")
    return {
        "style_proxy_cases": str(len(rows)),
        "total_style_proxy_wins": str(total_wins),
        "total_style_proxy_checks": str(total_checks),
        "overall_style_proxy_win_rate": format_float(total_wins / total_checks if total_checks else 0.0),
        "mean_case_style_proxy_win_rate": format_float(sum(rates) / len(rates) if rates else 0.0),
        "best_case": best["case"],
        "best_case_win_rate": format_float(as_float(best.get("style_proxy_win_rate")) or 0.0),
        "weakest_case": weakest["case"],
        "weakest_case_win_rate": format_float(as_float(weakest.get("style_proxy_win_rate")) or 0.0),
        "main_final_train_loss": format_float(main_train),
        "main_final_val_loss": format_float(main_val),
        "continued_final_train_loss": format_float(cont_train),
        "continued_final_val_loss": format_float(cont_val),
    }


def build_figure_pack(metrics_dir: Path, tables_dir: Path, figures_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(metrics_dir / "style_proxy_scores.csv")
    if not rows:
        raise ValueError(f"required input has no rows: {metrics_dir / 'style_proxy_scores.csv'}")

    plot_scorecard(rows, figures_dir / "eval_style_proxy_scorecard.png")
    plot_delta_heatmap(rows, figures_dir / "eval_metric_delta_heatmap.png")
    plot_distill_loss(
        tables_dir / "lora_train_val_loss.csv",
        tables_dir / "lora_cont_train_val_loss.csv",
        figures_dir / "eval_distill_loss_comparison.png",
    )

    summary = build_summary(rows, tables_dir)
    write_summary(tables_dir / "evaluation_summary.csv", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/metrics_lora_200_cont_lr5e5_seed7_styles"),
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("outputs/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    summary = build_figure_pack(args.metrics_dir, args.tables_dir, args.figures_dir)
    print(
        "evaluation figure pack: "
        f"{summary['total_style_proxy_wins']}/{summary['total_style_proxy_checks']} "
        f"wins, summary={args.tables_dir / 'evaluation_summary.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
