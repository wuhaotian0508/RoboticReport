#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-./MoConVQ/.venv/Scripts/python.exe}"

echo "[1/5] Build or diagnose style subset"
"$PYTHON" scripts/filter_style_subset.py --allow-missing

echo "[2/5] Create small style splits"
"$PYTHON" scripts/make_small_splits.py || true

echo "[3/5] Write fine-tuning diagnostic"
"$PYTHON" scripts/train_style_adapter.py --fp16

echo "[4/5] Verify midterm artifacts"
"$PYTHON" scripts/verify_midterm_artifacts.py

echo "[5/5] Baseline generation command preview"
"$PYTHON" scripts/run_baseline_prompts.py --dry-run

echo "Midterm pipeline check finished."
