# MoConVQ Style-Adapter Fine-Tuning Guide

This project uses LoRA style-caption distillation as the final practical
adaptation method.

The accepted proposal planned to fine-tune MoConVQ's language-conditioning
transformer using HumanML3D motion supervision. The available HumanML3D parquet
release contains processed 263-D motion features, not MoConVQ-compatible BVH
files or discrete motion-token labels. We therefore use pretrained MoConVQ as a
teacher to create pseudo token targets for style captions, then train only
low-rank LoRA updates in the frozen text-conditioned generator.

## 1. Environment

```powershell
conda create -n roboticsreport_lora python=3.8 pip -y
D:\anaconda3\envs\roboticsreport_lora\python.exe -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
D:\anaconda3\envs\roboticsreport_lora\python.exe -m pip install einops==0.6.0 h5py==3.8.0 matplotlib==3.7.1 scikit-learn==1.2.2 scipy==1.10.1 tqdm==4.65.0 setuptools==58.2.0 tensorboardx opt_einsum numba psutil pyyaml cython==0.29.36 transformers==4.41.2 sentencepiece pandas pyarrow opencv-python imageio imageio-ffmpeg
```

Use the explicit interpreter path to avoid Windows `conda run` Unicode logging
issues.

## 2. Build Style Subsets

```powershell
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\filter_style_subset.py --humanml3d-root HumanML3D
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\make_small_splits.py --train-size 200 --val-size 40 --test-size 40
```

Expected files:

```text
data/style_subset/style_subset_train.csv
data/style_subset/style_subset_val.csv
data/style_subset/style_subset_test.csv
data/style_subset/style_subset_statistics.csv
data/style_subset_small/style_train_small.csv
```

## 3. Smoke Test

```powershell
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\train_style_lora_distill.py `
  --output-dir outputs\finetune_lora_smoke `
  --cache-path outputs\finetune_distill_smoke\distill_cache.pt `
  --train-only `
  --epochs 1 `
  --log-every 1 `
  --lora-rank 4 `
  --lora-alpha 8
```

Measured smoke output:

```text
outputs/finetune_lora_smoke/train_log.csv
outputs/finetune_lora_smoke/style_lora_last.pth
```

## 4. Main LoRA Run

```powershell
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\train_style_lora_distill.py `
  --output-dir outputs\finetune_lora_200 `
  --cache-path outputs\finetune_lora_200\distill_cache.pt `
  --max-samples 200 `
  --max-length 50 `
  --build-cache-only `
  --rebuild-cache

D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\train_style_lora_distill.py `
  --output-dir outputs\finetune_lora_200 `
  --cache-path outputs\finetune_lora_200\distill_cache.pt `
  --train-only `
  --epochs 5 `
  --lora-rank 8 `
  --lora-alpha 16 `
  --lr 1e-4
```

Measured outputs:

```text
outputs/finetune_lora_200/distill_cache.pt
outputs/finetune_lora_200/train_log.csv
outputs/finetune_lora_200/config.json
outputs/finetune_lora_200/style_lora_epoch0.pth ... style_lora_epoch4.pth
outputs/finetune_lora_200/style_lora_last.pth
```

## 5. Generate BVH with LoRA

```powershell
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\generate_with_style_lora.py `
  --checkpoint outputs\finetune_lora_200\style_lora_last.pth `
  --prompt-file data\prompts\baseline_and_style_prompts.txt `
  --output-dir outputs\finetune_lora_200_samples `
  --max-length 50
```

## 6. Summarize Report Artifacts

Render selected BVH files with `scripts/render_bvh_video.py`, then summarize:

```powershell
D:\anaconda3\envs\roboticsreport_lora\python.exe scripts\summarize_lora_results.py `
  --run-dir outputs\finetune_lora_200 `
  --comparison-dir outputs\comparison_lora_200
```

Expected report artifacts:

```text
outputs/tables/lora_training_summary.csv
outputs/figures/lora_loss_curve.png
outputs/figures/lora_qualitative_frames.png
outputs/comparison_lora_200/comparison_manifest.csv
```

## Report Wording

Use this description:

> Since the available HumanML3D parquet release provides processed motion
> features rather than MoConVQ-compatible BVH or token labels, we implement a
> parameter-efficient LoRA style-caption distillation pipeline. The pretrained
> MoConVQ text generator serves as a teacher to produce deterministic pseudo
> token targets for style-focused captions. The student keeps the pretrained
> generator frozen and trains only low-rank LoRA updates, preserving the
> proposal's frozen-backbone constraint while avoiding unsupported format
> conversion.

Do not claim official FID or R-Precision unless a compatible evaluator is added
and run.
