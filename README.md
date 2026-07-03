# Robotics Project Deliverables

This folder is the project layer built around the upstream `MoConVQ/` codebase.
The accepted proposal targets style-aware adaptation for language-conditioned
humanoid motion generation.

## What Is Included

- `proposal.pdf`: accepted proposal.
- `PROJECT_PLAN.md`: project execution plan.
- `DELIVERABLES.md`: current submission checklist.
- `scripts/`: reproducible tools for data curation, baseline generation, LoRA distillation, generation, rendering, train/validation loss evaluation, BVH proxy metrics, and summarization.
- `data/`: prompt lists, style lexicon, style-filtered HumanML3D metadata, and audit templates.
- `outputs/`: measured baseline, LoRA, rendered comparison, table, figure, and log outputs.
- `report/`: midterm and final LaTeX reports with shared references.

## Repository History Note

The `main` branch is the cleaned final submission branch. During earlier
repository synchronization, the first project commits on the original main
history were overwritten by a remote update. Those exact commits could not be
recovered reliably from the available repository references, so the closest
preserved and reordered version of that early work is kept on the
`wuhao-first-main` branch. Use `main` for the final submitted report and code
state; use `wuhao-first-main` only when checking the earlier development
history.

## Team Contributions

This contribution summary is based on the visible git commit history after the
repository-history cleanup described above.

- Haotian Wu (`wuhaotian0508` / `haotian wu`): led repository setup and final
  integration, including the initial project import, dataset exclusion rules,
  large checkpoint/LFS handling, final submission packaging, RSS-format report
  cleanup, GitHub history repair, and README documentation. He also led the
  final HumanML3D-guided teacher-selection extension, updated the final report
  and submission PDFs, prepared final presentation materials, and coordinated
  the cleaned `main` branch with the preserved `wuhao-first-main` history.
- Yuhua Luo (`silhovette` in the commit history): implemented and iterated on
  the LoRA distillation experiment pipeline, including the smoke pipeline,
  measured final LoRA runs, continued-refinement experiments, generated BVH
  samples, proxy metric evaluation, and paper-ready result figures. He also
  organized missing submission artifacts, expanded the midterm and final
  reports with experimental results and analysis, converted the reports to the
  RSS/IEEEtran format, refined the narrative around LoRA improvements, and
  prepared the final evaluation figure pack used in the paper.
- Xuyang Yuan (`Yuan Xuyang` in the commit history): contributed the human
  evaluation branch and related integration work, including human-evaluation
  additions, pairwise-evaluation material integration, conflict-marker cleanup,
  and pull-request/merge support for evaluation materials. He also helped
  maintain repository consistency during collaborative merging and contributed
  supporting project updates that were incorporated into the final cleaned
  branch.

## Current Method

The original proposal planned to fine-tune MoConVQ's language-conditioning
transformer using HumanML3D style motions while keeping the motion backbone
frozen. The available HumanML3D data in this workspace is the Hugging Face
parquet export with processed 263-D motion features, not MoConVQ-compatible BVH
files or discrete token labels.

The implemented final method is therefore a parameter-efficient backup:

1. Filter style-related HumanML3D captions without crossing train/val/test splits.
2. Use pretrained MoConVQ/MoConGPT as a teacher to generate pseudo token targets.
3. Freeze the pretrained generator and train only LoRA updates in the text-conditioned transformer.
4. Continue the selected LoRA adapter with a lower learning rate and generate deterministic style-prompt BVH outputs.
5. Report measured training loss, train/validation pseudo-token loss, BVH kinematic proxy metrics, and qualitative comparisons. Do not claim FID or R-Precision without a compatible evaluator.

## Conda Environment

The active reproducible environment is:

```powershell
conda create -n roboticsreport_lora python=3.8 pip -y
D:\anaconda3\envs\roboticsreport_lora\python.exe -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
D:\anaconda3\envs\roboticsreport_lora\python.exe -m pip install einops==0.6.0 h5py==3.8.0 matplotlib==3.7.1 scikit-learn==1.2.2 scipy==1.10.1 tqdm==4.65.0 setuptools==58.2.0 tensorboardx opt_einsum numba psutil pyyaml cython==0.29.36 transformers==4.41.2 sentencepiece pandas pyarrow opencv-python imageio imageio-ffmpeg
```

Use the explicit interpreter path in commands to avoid Windows `conda run`
encoding issues:

```powershell
D:\anaconda3\envs\roboticsreport_lora\python.exe ...
```

## Reproduction Order

```powershell
# Verify baseline deliverables
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\verify_midterm_artifacts.py

# Build style subsets from HumanML3D parquet export
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\filter_style_subset.py --humanml3d-root HumanML3D
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\make_small_splits.py --train-size 200 --val-size 40 --test-size 40

# Train LoRA adapter from an existing or rebuilt distillation cache
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\train_style_lora_distill.py --output-dir outputs\finetune_lora_200 --cache-path outputs\finetune_lora_200\distill_cache.pt --max-samples 200 --max-length 50 --build-cache-only --rebuild-cache
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\train_style_lora_distill.py --output-dir outputs\finetune_lora_200 --cache-path outputs\finetune_lora_200\distill_cache.pt --train-only --epochs 5 --lora-rank 8 --lora-alpha 16 --lr 1e-4

# Generate adapted BVH outputs
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\generate_with_style_lora.py --checkpoint outputs\finetune_lora_200\style_lora_last.pth --prompt-file data\prompts\baseline_and_style_prompts.txt --output-dir outputs\finetune_lora_200_samples --max-length 50

# Optional final refinement selected for the report
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\train_style_lora_distill.py --output-dir outputs\finetune_lora_200_cont_lr5e5 --cache-path outputs\finetune_lora_200\distill_cache.pt --train-only --epochs 2 --lora-rank 8 --lora-alpha 16 --lr 5e-5 --resume-lora outputs\finetune_lora_200\style_lora_epoch4.pth
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\generate_with_style_lora.py --checkpoint outputs\finetune_lora_200_cont_lr5e5\style_lora_last.pth --prompt-file data\prompts\baseline_and_style_prompts.txt --output-dir outputs\finetune_lora_200_cont_lr5e5_seed7_styles --max-length 50 --seed 7 --start-index 5 --end-index 9

# Render selected baseline/LoRA comparisons and summarize figures/tables
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\summarize_lora_results.py --run-dir outputs\finetune_lora_200 --comparison-dir outputs\comparison_lora_200_cont

# Evaluate checkpoint-level pseudo-token loss and BVH style proxy metrics
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\evaluate_lora_distill_loss.py --checkpoint-dir outputs\finetune_lora_200 --train-cache outputs\finetune_lora_200\distill_cache.pt --val-cache outputs\finetune_lora_val\distill_cache.pt --output outputs\tables\lora_train_val_loss.csv
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\evaluate_lora_distill_loss.py --checkpoint-dir outputs\finetune_lora_200_cont_lr5e5 --train-cache outputs\finetune_lora_200\distill_cache.pt --val-cache outputs\finetune_lora_val\distill_cache.pt --output outputs\tables\lora_cont_train_val_loss.csv
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\compute_bvh_proxy_metrics.py --baseline-dir outputs\baseline\project_prompts --lora-dir outputs\finetune_lora_200_cont_lr5e5_seed7_styles --output-dir outputs\metrics_lora_200_cont_lr5e5_seed7_styles

# Build paper-ready evaluation figure pack
uv run --python .\MoConVQ\.venv\Scripts\python.exe python scripts\make_evaluation_figure_pack.py --metrics-dir outputs\metrics_lora_200_cont_lr5e5_seed7_styles --tables-dir outputs\tables --figures-dir outputs\figures
```

## Report Compilation

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compile_reports.ps1
```

Compiled PDFs are written next to their `.tex` sources if LaTeX succeeds.

## Integrity Note

This workspace reports only measured results: HumanML3D style subset counts,
baseline BVH outputs, LoRA distillation loss, continued-LoRA refinement loss,
checkpoint-level train/validation pseudo-token loss, LoRA checkpoints, generated BVHs, BVH style proxy metrics,
and rendered qualitative comparisons. It does not report official FID or
R-Precision because the current data/model formats are not compatible with that
evaluation without additional conversion or an official evaluator bridge.
