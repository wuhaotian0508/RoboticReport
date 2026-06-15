from __future__ import annotations

import importlib.util
import tempfile
import unittest
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "humanml_motion_proxy.py"


def load_module():
    if not SCRIPT.exists():
        raise AssertionError(f"missing script: {SCRIPT}")
    spec = importlib.util.spec_from_file_location("humanml_motion_proxy", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HumanMLMotionProxyTests(unittest.TestCase):
    def test_computes_motion_metrics_from_humanml_263_features(self) -> None:
        module = load_module()
        motion = np.zeros((4, 263), dtype=np.float32)
        motion[:, 1] = [1.0, 1.0, 1.0, 1.0]
        motion[:, 2] = [0.0, 0.0, 0.0, 0.0]
        motion[:, 3] = [0.5, 0.6, 0.4, 0.7]
        motion[:, 5] = [0.1, 0.2, 0.3, 0.4]
        motion[:, 8] = [0.4, 0.3, 0.2, 0.1]

        metrics = module.compute_humanml_metrics(motion, fps=20.0)

        self.assertEqual(metrics["frames"], 4.0)
        self.assertAlmostEqual(metrics["duration_sec"], 0.2)
        self.assertAlmostEqual(metrics["root_speed_mean"], 1.0)
        self.assertAlmostEqual(metrics["root_path_length"], 0.2)
        self.assertAlmostEqual(metrics["root_vertical_range"], 0.3)
        self.assertGreater(metrics["mean_joint_velocity"], 0.0)

    def test_selects_candidate_closest_to_humanml_motion_profile(self) -> None:
        module = load_module()
        target = {
            "root_speed_mean": 0.25,
            "mean_joint_velocity": 0.5,
            "root_vertical_range": 0.1,
            "duration_sec": 2.0,
        }
        candidates = [
            {"name": "fast", "metrics": {"root_speed_mean": 2.0, "mean_joint_velocity": 2.0, "root_vertical_range": 1.0, "duration_sec": 0.5}},
            {"name": "near", "metrics": {"root_speed_mean": 0.3, "mean_joint_velocity": 0.55, "root_vertical_range": 0.12, "duration_sec": 1.9}},
        ]

        selected = module.select_best_candidate(candidates, target)

        self.assertEqual(selected["name"], "near")
        self.assertLess(selected["selection_score"], candidates[0]["selection_score"])

    def test_motion_store_loads_matching_parquet_motion(self) -> None:
        module = load_module()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "train-00000-of-00001.parquet"
            wanted_motion = np.ones((3, 263), dtype=np.float32).tolist()
            other_motion = np.zeros((3, 263), dtype=np.float32).tolist()
            table = pa.Table.from_pylist(
                [
                    {"caption": "other", "motion": other_motion, "meta_data": {"name": "000000", "duration": 1.0, "num_frames": 3}},
                    {"caption": "wanted", "motion": wanted_motion, "meta_data": {"name": "000001", "duration": 1.0, "num_frames": 3}},
                ]
            )
            pq.write_table(table, parquet_path)

            store = module.HumanMLMotionStore()
            motion = store.load_motion(parquet_path, "000001")

            self.assertEqual(motion.shape, (3, 263))
            self.assertAlmostEqual(float(motion.mean()), 1.0)

    def test_reads_style_rows_with_motion_metadata(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "style.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["caption", "motion_id", "text_path", "style_groups", "matched_keywords"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "caption": "slow walk",
                        "motion_id": "000001",
                        "text_path": "HumanML3D/data/train.parquet",
                        "style_groups": "Controlled motion",
                        "matched_keywords": "slow",
                    }
                )

            rows = module.read_style_rows(csv_path, max_samples=10, seed=7)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["caption"], "slow walk")
            self.assertEqual(rows[0]["motion_id"], "000001")
            self.assertEqual(rows[0]["text_path"], "HumanML3D/data/train.parquet")

    def test_make_teacher_example_passes_sampling_mode_to_teacher(self) -> None:
        import torch

        script = ROOT / "scripts" / "train_style_lora_distill.py"
        spec = importlib.util.spec_from_file_location("train_style_lora_distill_for_test", script)
        if spec is None or spec.loader is None:
            raise AssertionError(f"could not load script: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        calls = []

        class Teacher:
            def sample(self, *args, **kwargs):
                calls.append(kwargs["if_categorial"])
                return torch.ones((1, 3, 768)), torch.zeros((12, 1), dtype=torch.long)

        module.text2bert = lambda texts, tokenizer, encoder, device: (
            torch.zeros((1, 2, 768)),
            torch.zeros((1, 2), dtype=torch.bool),
        )

        example = module.make_teacher_example(
            Teacher(),
            "a person walks slowly",
            tokenizer=None,
            encoder=None,
            device=torch.device("cpu"),
            max_length=5,
            teacher_samples=2,
            teacher_categorical_sampling=True,
        )

        self.assertIsNotNone(example)
        self.assertEqual(calls, [True, True])


if __name__ == "__main__":
    unittest.main()
