from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_evaluation_figure_pack.py"


def load_module():
    if not SCRIPT.exists():
        raise AssertionError(f"missing script: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("make_evaluation_figure_pack", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EvaluationFigurePackTests(unittest.TestCase):
    def test_builds_summary_and_all_figures(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            tables_dir = root / "tables"
            figures_dir = root / "figures"
            write_csv(
                metrics_dir / "style_proxy_scores.csv",
                [
                    {
                        "case": "slow_deliberate",
                        "prompt": "A person walks slowly.",
                        "style_proxy_wins": 3,
                        "style_proxy_total": 4,
                        "style_proxy_win_rate": 0.75,
                        "root_speed_mean_delta_pct": 12.5,
                        "mean_joint_jerk_delta_pct": 8.0,
                    },
                    {
                        "case": "tai_chi",
                        "prompt": "A person moves like doing tai chi.",
                        "style_proxy_wins": 1,
                        "style_proxy_total": 4,
                        "style_proxy_win_rate": 0.25,
                        "duration_sec_delta_pct": -5.0,
                        "root_rotation_amount_delta_pct": 6.0,
                    },
                ],
            )
            write_csv(
                tables_dir / "lora_train_val_loss.csv",
                [
                    {
                        "checkpoint": "style_lora_epoch0.pth",
                        "epoch": 0,
                        "train_examples": 200,
                        "train_loss": 0.1,
                        "val_examples": 40,
                        "val_loss": 0.2,
                    }
                ],
            )
            write_csv(
                tables_dir / "lora_cont_train_val_loss.csv",
                [
                    {
                        "checkpoint": "style_lora_epoch1.pth",
                        "epoch": 1,
                        "train_examples": 200,
                        "train_loss": 0.03,
                        "val_examples": 40,
                        "val_loss": 0.04,
                    }
                ],
            )

            summary = module.build_figure_pack(metrics_dir, tables_dir, figures_dir)

            self.assertEqual(summary["total_style_proxy_wins"], "4")
            self.assertEqual(summary["total_style_proxy_checks"], "8")
            self.assertEqual(summary["best_case"], "slow_deliberate")
            self.assertEqual(summary["weakest_case"], "tai_chi")
            self.assertEqual(summary["continued_final_val_loss"], "0.040000")

            for name in (
                "eval_style_proxy_scorecard.png",
                "eval_metric_delta_heatmap.png",
                "eval_distill_loss_comparison.png",
            ):
                image = figures_dir / name
                self.assertTrue(image.exists(), name)
                self.assertGreater(image.stat().st_size, 1000, name)

            summary_path = tables_dir / "evaluation_summary.csv"
            self.assertTrue(summary_path.exists())
            with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["overall_style_proxy_win_rate"], "0.500000")

    def test_missing_required_style_scores_names_path(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "metrics"
            tables_dir = root / "tables"
            figures_dir = root / "figures"
            with self.assertRaisesRegex(FileNotFoundError, "style_proxy_scores.csv"):
                module.build_figure_pack(metrics_dir, tables_dir, figures_dir)


if __name__ == "__main__":
    unittest.main()
