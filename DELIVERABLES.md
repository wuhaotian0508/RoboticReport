# Deliverables

## Batch 1: Midterm Submission

- [x] RSS-style report source: `report/midterm/midterm.tex`
- [x] Shared bibliography: `report/shared/references.bib`
- [x] Pipeline figure embedded with TikZ in the report
- [x] Implementation progress table in the report
- [x] Style subset script: `scripts/filter_style_subset.py`
- [x] Small split script: `scripts/make_small_splits.py`
- [x] Baseline prompt list: `data/prompts/baseline_and_style_prompts.txt`
- [x] Manual audit template: `data/manual_audit_template.csv`
- [x] Baseline generation entry point: `scripts/run_baseline_prompts.py`
- [x] Fine-tuning entry point: `scripts/train_style_adapter.py`
- [x] Compile helper: `scripts/compile_reports.ps1`
- [x] Midterm status log: `outputs/logs/midterm_status.md`
- [x] Baseline verification helper: `scripts/verify_midterm_artifacts.py`

## Batch 2: Final Submission

- [x] Final report source: `report/final/final_report.tex`
- [x] Reproducible data curation pipeline
- [x] Reproducible baseline generation pipeline
- [x] Parameter-efficient fine-tuning pipeline
- [x] Metric script for FID and R-Precision from extracted features
- [x] Output directory structure:
  - `outputs/baseline/`
  - `outputs/finetune/`
  - `outputs/figures/`
  - `outputs/logs/`
- [x] Clear external-asset checklist

## Items To Fill After Full Experiment Run

- [ ] Real style subset statistics from HumanML3D.
- [x] Baseline BVH outputs for at least ten prompts.
- [ ] Fine-tuning loss CSV and checkpoint.
- [ ] FID and R-Precision values on full and style-specific test sets.
- [ ] Side-by-side qualitative comparison frames.
