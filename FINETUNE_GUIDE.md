# MoConVQ Style-Adapter Fine-Tuning Guide

This project uses a practical style-caption distillation setup:

1. Filter style-related HumanML3D captions.
2. Use pretrained MoConGPT as a teacher to generate deterministic token targets.
3. Fine-tune a student MoConGPT on those caption-token pairs.
4. Generate BVH motions from the adapted checkpoint.

This is a preliminary adaptation experiment because the downloaded HumanML3D parquet data contains processed 263-D motion features, not MoConVQ-compatible BVH files.

## 1. Build Style Subsets

```bash
cd /d/roboticsreport

./MoConVQ/.venv/Scripts/python.exe scripts/filter_style_subset.py \
  --humanml3d-root D:/roboticsreport/HumanML3D

./MoConVQ/.venv/Scripts/python.exe scripts/make_small_splits.py \
  --train-size 200 --val-size 40 --test-size 40
```

Expected files:

```text
data/style_subset/style_subset_train.csv
data/style_subset/style_subset_val.csv
data/style_subset/style_subset_test.csv
data/style_subset/style_subset_statistics.csv
data/style_subset_small/style_train_small.csv
```

## 2. Smoke Test Training

```bash
export HF_ENDPOINT=https://hf-mirror.com
export NO_PROXY='*'
export no_proxy='*'

./MoConVQ/.venv/Scripts/python.exe scripts/train_style_adapter_distill.py \
  --train-csv data/style_subset_small/style_train_small.csv \
  --output-dir outputs/finetune_distill_smoke \
  --cache-path outputs/finetune_distill_smoke/distill_cache.pt \
  --max-samples 2 \
  --epochs 1 \
  --max-length 12 \
  --log-every 1 \
  --rebuild-cache
```

## 3. Small Experiment

```bash
./MoConVQ/.venv/Scripts/python.exe scripts/train_style_adapter_distill.py \
  --train-csv data/style_subset_small/style_train_small.csv \
  --output-dir outputs/finetune_distill \
  --cache-path outputs/finetune_distill/distill_cache.pt \
  --max-samples 20 \
  --epochs 2 \
  --max-length 30 \
  --log-every 5 \
  --rebuild-cache
```

Expected files:

```text
outputs/finetune_distill/train_log.csv
outputs/finetune_distill/distill_cache.pt
outputs/finetune_distill/style_adapter_epoch0.pth
outputs/finetune_distill/style_adapter_epoch1.pth
outputs/finetune_distill/style_adapter_last.pth
```

## 4. Larger Run

After the small experiment works, increase sample count and epochs:

```bash
./MoConVQ/.venv/Scripts/python.exe scripts/train_style_adapter_distill.py \
  --train-csv data/style_subset_small/style_train_small.csv \
  --output-dir outputs/finetune_distill_200 \
  --cache-path outputs/finetune_distill_200/distill_cache.pt \
  --max-samples 200 \
  --epochs 5 \
  --max-length 50 \
  --lr 5e-6 \
  --log-every 10 \
  --rebuild-cache
```

If a cache already exists and you only want to retrain from it, omit `--rebuild-cache`.

## 5. Generate BVH with the Adapter

```bash
./MoConVQ/.venv/Scripts/python.exe scripts/generate_with_style_adapter.py \
  --checkpoint outputs/finetune_distill/style_adapter_last.pth \
  --prompt "A person moves like doing tai chi." \
  --output-dir outputs/finetune_distill_samples \
  --max-length 30
```

For multiple prompts:

```bash
./MoConVQ/.venv/Scripts/python.exe scripts/generate_with_style_adapter.py \
  --checkpoint outputs/finetune_distill/style_adapter_last.pth \
  --prompt-file data/prompts/baseline_and_style_prompts.txt \
  --output-dir outputs/finetune_distill_samples \
  --max-length 50
```

## Report Wording

Use this description:

> Since the downloaded HumanML3D parquet release provides processed motion features rather than MoConVQ-compatible BVH files, we implement a style-caption distillation fine-tuning pipeline. The pretrained MoConGPT serves as a teacher to produce deterministic token targets for style-focused captions, and the student generator is optimized to reproduce these token sequences. This provides a preliminary parameter-efficient adaptation experiment while preserving true HumanML3D token supervision as future work.

Do not claim FID or R-Precision unless separately measured.
