# Midterm Status

Checked on 2026-05-21.

## Completed

- MoConVQ pretrained assets are present in `MoConVQ/`.
- Motion tokenization and decoding have been verified.
- Default text-to-motion inference has been verified.
- Ten project prompts generated BVH files under `MoConVQ/out/project_prompts/`.
- The same ten prompt BVH files are copied to `outputs/baseline/project_prompts/` for midterm evidence.
- Baseline reproduction details are copied to `outputs/logs/baseline_reproduction_log.md`.
- Split-preserving style-subset tooling has been implemented under `scripts/`.
- Fine-tuning is represented by an honest diagnostic entry point until token targets exist.

## Pending

- HumanML3D is expected at `D:\roboticsreport\datasets\HumanML3D`.
- Style subset statistics are placeholders until HumanML3D `texts/`, `train.txt`, `val.txt`, and `test.txt` are available.
- Fine-tuning requires tokenized MoConVQ target sequences for the selected HumanML3D motions.
- FID and R-Precision have not been measured.

## Git Bash Entry Point

```bash
cd /d/roboticsreport
./MoConVQ/.venv/Scripts/python.exe scripts/verify_midterm_artifacts.py
./MoConVQ/.venv/Scripts/python.exe scripts/filter_style_subset.py --allow-missing
./MoConVQ/.venv/Scripts/python.exe scripts/train_style_adapter.py --fp16
```
