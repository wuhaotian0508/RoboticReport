# Robotics Project Deliverables

This folder is the project layer built around the upstream `MoConVQ/` codebase.

## What Is Included

- `PROJECT_PLAN.md`: complete project execution plan.
- `DELIVERABLES.md`: submission checklist for midterm and final batches.
- `scripts/`: reproducible command-line tools for data curation, generation, fine-tuning, and evaluation.
- `data/`: prompt lists, style lexicon, audit template, and example metadata.
- `outputs/`: output directories and status logs.
- `report/`: midterm and final LaTeX reports with shared references.

## External Files Required for Full Experiments

Place official MoConVQ pretrained files in `MoConVQ/`:

- `moconvq_base.data`
- `text_generation_GPT.pth`

Prepare HumanML3D annotations and split files at the fixed project path:

```text
D:/roboticsreport/datasets/HumanML3D/
  texts/
    000000.txt
    ...
  train.txt
  val.txt
  test.txt
```

The scripts default to this root, but `--humanml3d-root` can point elsewhere.

## Minimal Run Order

```bash
cd /d/roboticsreport

./MoConVQ/.venv/Scripts/python.exe scripts/verify_midterm_artifacts.py
./MoConVQ/.venv/Scripts/python.exe scripts/filter_style_subset.py --allow-missing
./MoConVQ/.venv/Scripts/python.exe scripts/make_small_splits.py || true
./MoConVQ/.venv/Scripts/python.exe scripts/train_style_adapter.py --fp16
```

Or run the same midterm checks in one Git Bash command:

```bash
cd /d/roboticsreport
bash scripts/run_midterm_pipeline.sh
```

## Report Compilation

```bash
cd /d/roboticsreport
powershell -ExecutionPolicy Bypass -File scripts/compile_reports.ps1
```

Compiled PDFs are written next to their `.tex` sources if LaTeX succeeds.

## Integrity Note

This workspace currently has the MoConVQ pretrained weights and baseline BVH outputs, but lacks HumanML3D. The reports therefore use a diagnostic-progress framing and do not invent FID, R-Precision, or fine-tuning results. After adding HumanML3D, rerun the scripts and replace pending tables with measured outputs.
