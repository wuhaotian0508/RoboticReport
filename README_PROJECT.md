# Robotics Project Deliverables

This folder is the project layer built around the upstream `MoConVQ/` codebase.
The accepted proposal targets style-aware adaptation for language-conditioned
humanoid motion generation.

## What Is Included

- `proposal.pdf`: accepted proposal.
- `PROJECT_PLAN.md`: project execution plan.
- `DELIVERABLES.md`: current submission checklist.
- `scripts/`: reproducible tools for data curation, baseline generation, LoRA distillation, generation, rendering, and summarization.
- `data/`: prompt lists, style lexicon, style-filtered HumanML3D metadata, and audit templates.
- `outputs/`: measured baseline, LoRA, rendered comparison, table, figure, and log outputs.
- `report/`: midterm and final LaTeX reports with shared references.

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
4. Generate BVH outputs for the same prompts as the baseline.
5. Report measured training loss and qualitative comparisons. Do not claim FID or R-Precision without a compatible evaluator.

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

# Render selected baseline/LoRA comparisons and summarize figures/tables
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\summarize_lora_results.py --run-dir outputs\finetune_lora_200 --comparison-dir outputs\comparison_lora_200
```

## Report Compilation

```powershell
powershell -ExecutionPolicy Bypass -File scripts\compile_reports.ps1
```

Compiled PDFs are written next to their `.tex` sources if LaTeX succeeds.

## Integrity Note

This workspace reports only measured results: HumanML3D style subset counts,
baseline BVH outputs, LoRA distillation loss, LoRA checkpoints, generated BVHs,
and rendered qualitative comparisons. It does not report official FID or
R-Precision because the current data/model formats are not compatible with that
evaluation without additional conversion or an official evaluator bridge.
