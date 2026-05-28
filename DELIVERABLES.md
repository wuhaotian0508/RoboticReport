# Deliverables

## Batch 1: Midterm Submission

- [x] Strict RSS/IEEEtran report source: `report/midterm/midterm.tex`
- [x] RSS template class copied for reproducible compilation: `report/midterm/IEEEtran.cls`
- [x] Shared bibliography: `report/shared/references.bib`
- [x] Pipeline figure in the report
- [x] Implementation progress table in the report
- [x] Style subset script: `scripts/filter_style_subset.py`
- [x] Small split script: `scripts/make_small_splits.py`
- [x] Baseline prompt list: `data/prompts/baseline_and_style_prompts.txt`
- [x] Manual audit template: `data/manual_audit_template.csv`
- [x] Baseline generation entry point: `scripts/run_baseline_prompts.py`
- [x] Original diagnostic fine-tuning entry point: `scripts/train_style_adapter.py`
- [x] Compile helper: `scripts/compile_reports.ps1`
- [x] Midterm status log: `outputs/logs/midterm_status.md`
- [x] Baseline verification helper: `scripts/verify_midterm_artifacts.py`

## Batch 2: Final Submission

- [x] Final report source: `report/final/final_report.tex`
- [x] RSS template class copied for reproducible compilation: `report/final/IEEEtran.cls`
- [x] Reproducible data curation pipeline
- [x] Reproducible baseline generation pipeline
- [x] Parameter-efficient LoRA distillation pipeline:
  - `scripts/style_lora.py`
  - `scripts/train_style_lora_distill.py`
  - `scripts/generate_with_style_lora.py`
- [x] LoRA result summarizer: `scripts/summarize_lora_results.py`
- [x] BVH proxy metric script: `scripts/compute_bvh_proxy_metrics.py`
- [x] LoRA train/validation loss evaluator: `scripts/evaluate_lora_distill_loss.py`
- [x] Output directory structure:
  - `outputs/baseline/`
  - `outputs/finetune_lora_200/`
  - `outputs/finetune_lora_200_cont_lr5e5/`
  - `outputs/finetune_lora_200_samples/`
  - `outputs/finetune_lora_200_cont_lr5e5_seed7_styles/`
  - `outputs/comparison_lora_200/`
  - `outputs/comparison_lora_200_cont/`
  - `outputs/figures/`
  - `outputs/logs/`
- [x] Clear external-asset and environment checklist

## Completed Experiment Artifacts

- [x] Real style subset statistics from HumanML3D parquet export:
  - `data/style_subset/style_subset_statistics.csv`
- [x] Baseline BVH outputs for ten prompts:
  - `outputs/baseline/project_prompts/`
- [x] LoRA fine-tuning loss CSV and checkpoints:
  - `outputs/finetune_lora_200/train_log.csv`
  - `outputs/finetune_lora_200/style_lora_last.pth`
- [x] Continued LoRA refinement run:
  - `outputs/finetune_lora_200_cont_lr5e5/train_log.csv`
  - `outputs/finetune_lora_200_cont_lr5e5/style_lora_last.pth`
- [x] LoRA-generated BVH outputs for ten prompts:
  - `outputs/finetune_lora_200_samples/`
- [x] Selected continued-LoRA style prompt BVH outputs:
  - `outputs/finetune_lora_200_cont_lr5e5_seed7_styles/`
- [x] Side-by-side qualitative comparison videos and figure:
  - `outputs/comparison_lora_200_cont/`
  - `outputs/figures/lora_qualitative_frames.png`
- [x] Loss curve and epoch summary:
  - `outputs/figures/lora_loss_curve.png`
  - `outputs/figures/lora_refinement_loss.png`
  - `outputs/tables/lora_training_summary.csv`
  - `outputs/tables/lora_refinement_loss.csv`
- [x] Final-report diagnostic figures:
  - `outputs/figures/method_pipeline_diagram.png`
  - `outputs/figures/baseline_lora_key_metrics.png`
  - `outputs/figures/style_subset_distribution.png`
  - `outputs/figures/style_proxy_win_rate.png`
  - `outputs/figures/style_proxy_delta_heatmap.png`
- [x] BVH motion-style proxy metrics:
  - `outputs/metrics_lora_200/bvh_proxy_metrics.csv`
  - `outputs/metrics_lora_200/bvh_proxy_comparison.csv`
  - `outputs/metrics_lora_200/style_proxy_scores.csv`
  - `outputs/metrics_lora_200_cont_lr5e5_seed7_styles/style_proxy_scores.csv`
- [x] Continued-LoRA train/validation loss:
  - `outputs/tables/lora_cont_train_val_loss.csv`

## Not Claimed

- [ ] FID and R-Precision on the official HumanML3D evaluator.
Reason: the available HumanML3D parquet data contains processed 263-D motion features, while MoConVQ fine-tuning and generation operate on MoConVQ discrete motion tokens and BVH/controller outputs. This workspace therefore reports measured LoRA distillation loss and qualitative comparisons, but does not fabricate incompatible FID or R-Precision values.
