from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compute_humanml_alignment_metrics.py"


def load_module():
    if not SCRIPT.exists():
        raise AssertionError(f"missing script: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("compute_humanml_alignment_metrics", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HumanMLAlignmentMetricTests(unittest.TestCase):
    def test_alignment_rewards_closeness_not_extreme_lower_values(self) -> None:
        module = load_module()
        target = {"root_speed_mean": 0.5, "mean_joint_velocity": 1.0}
        close = {"root_speed_mean": 0.45, "mean_joint_velocity": 1.1}
        stopped = {"root_speed_mean": 0.0, "mean_joint_velocity": 0.0}

        close_distance = module.alignment_distance(close, target, ("root_speed_mean", "mean_joint_velocity"))
        stopped_distance = module.alignment_distance(stopped, target, ("root_speed_mean", "mean_joint_velocity"))

        self.assertLess(close_distance, stopped_distance)
        self.assertGreater(module.alignment_score(close_distance), module.alignment_score(stopped_distance))

    def test_compare_alignment_marks_lora_win_when_closer_to_humanml(self) -> None:
        module = load_module()
        target = {"root_speed_mean": 0.5, "duration_sec": 2.0}
        baseline = {"root_speed_mean": 1.0, "duration_sec": 1.0}
        lora = {"root_speed_mean": 0.55, "duration_sec": 2.1}

        row = module.compare_alignment(
            row_index=3,
            caption="slow walk",
            motion_id="000003",
            target_metrics=target,
            generated_metrics={"baseline": baseline, "lora": lora},
            metric_names=("root_speed_mean", "duration_sec"),
        )

        self.assertEqual(row["row_index"], 3)
        self.assertEqual(row["motion_id"], "000003")
        self.assertEqual(row["lora_closer_to_humanml"], 1)
        self.assertLess(row["lora_alignment_distance"], row["baseline_alignment_distance"])


if __name__ == "__main__":
    unittest.main()
