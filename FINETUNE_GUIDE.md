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
conda activate roboticsreport_lora
python -m pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install einops==0.6.0 h5py==3.8.0 matplotlib==3.7.1 scikit-learn==1.2.2 scipy==1.10.1 tqdm==4.65.0 setuptools==58.2.0 tensorboardx opt_einsum numba psutil pyyaml cython==0.29.36 transformers==4.41.2 sentencepiece pandas pyarrow opencv-python imageio imageio-ffmpeg
```

The commands below use `<env-python>` to denote the Python interpreter from this
environment. After activation, it can usually be replaced with `python`.

## 2. Build Style Subsets

```powershell
<env-python> scripts\filter_style_subset.py --humanml3d-root HumanML3D
<env-python> scripts\make_small_splits.py --train-size 200 --val-size 40 --test-size 40
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
<env-python> scripts\train_style_lora_distill.py `
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
<env-python> scripts\train_style_lora_distill.py `
  --output-dir outputs\finetune_lora_200 `
  --cache-path outputs\finetune_lora_200\distill_cache.pt `
  --max-samples 200 `
  --max-length 50 `
  --build-cache-only `
  --rebuild-cache

<env-python> scripts\train_style_lora_distill.py `
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
<env-python> scripts\generate_with_style_lora.py `
  --checkpoint outputs\finetune_lora_200\style_lora_last.pth `
  --prompt-file data\prompts\baseline_and_style_prompts.txt `
  --output-dir outputs\finetune_lora_200_samples `
  --max-length 50
```

## 6. Continued Refinement Used in the Final Report

The final reported proxy metrics use a conservative two-epoch continuation from
the epoch-4 checkpoint with a lower learning rate:

```powershell
<env-python> scripts\train_style_lora_distill.py `
  --output-dir outputs\finetune_lora_200_cont_lr5e5 `
  --cache-path outputs\finetune_lora_200\distill_cache.pt `
  --train-only `
  --epochs 2 `
  --lora-rank 8 `
  --lora-alpha 16 `
  --lr 5e-5 `
  --resume-lora outputs\finetune_lora_200\style_lora_epoch4.pth

<env-python> scripts\generate_with_style_lora.py `
  --checkpoint outputs\finetune_lora_200_cont_lr5e5\style_lora_last.pth `
  --prompt-file data\prompts\baseline_and_style_prompts.txt `
  --output-dir outputs\finetune_lora_200_cont_lr5e5_seed7_styles `
  --max-length 50 `
  --seed 7 `
  --start-index 5 `
  --end-index 9
```

This improves the style proxy score from 15/22 to 18/22 on the five style
prompts. This is still a proxy metric, not an official HumanML3D benchmark.

## 7. Summarize Report Artifacts

Render selected BVH files with `scripts/render_bvh_video.py`, then summarize:

```powershell
<env-python> scripts\summarize_lora_results.py `
  --run-dir outputs\finetune_lora_200 `
  --comparison-dir outputs\comparison_lora_200_cont
```

Expected report artifacts:

```text
outputs/tables/lora_training_summary.csv
outputs/figures/lora_loss_curve.png
outputs/figures/lora_qualitative_frames.png
outputs/comparison_lora_200_cont/comparison_manifest.csv
```

## 8. Quantitative Proxy Evaluation

Measure whether the adapter learned the teacher pseudo-token distribution:

```powershell
<env-python> scripts\evaluate_lora_distill_loss.py `
  --checkpoint-dir outputs\finetune_lora_200 `
  --train-cache outputs\finetune_lora_200\distill_cache.pt `
  --val-cache outputs\finetune_lora_val\distill_cache.pt `
  --output outputs\tables\lora_train_val_loss.csv
```

<env-python> scripts\evaluate_lora_distill_loss.py `
  --checkpoint-dir outputs\finetune_lora_200_cont_lr5e5 `
  --train-cache outputs\finetune_lora_200\distill_cache.pt `
  --val-cache outputs\finetune_lora_val\distill_cache.pt `
  --output outputs\tables\lora_cont_train_val_loss.csv

Measure BVH style-direction proxy metrics for baseline and LoRA outputs:

```powershell
<env-python> scripts\compute_bvh_proxy_metrics.py `
  --baseline-dir outputs\baseline\project_prompts `
  --lora-dir outputs\finetune_lora_200_cont_lr5e5_seed7_styles `
  --output-dir outputs\metrics_lora_200_cont_lr5e5_seed7_styles
```

Measured final proxy result:

```text
outputs/metrics_lora_200_cont_lr5e5_seed7_styles/style_proxy_scores.csv
total LoRA wins: 18 / 22 predefined style-direction comparisons
```

This is a project-specific proxy evaluation, not official HumanML3D FID or
R-Precision.

## 8.1 Paper-Ready Evaluation Figure Pack

Generate the final report figures and one-row evaluation summary from the
measured proxy/loss CSV files:

```powershell
python scripts\make_evaluation_figure_pack.py `
  --metrics-dir outputs\metrics_lora_200_cont_lr5e5_seed7_styles `
  --tables-dir outputs\tables `
  --figures-dir outputs\figures
```

Expected outputs:

```text
outputs/figures/eval_style_proxy_scorecard.png
outputs/figures/eval_metric_delta_heatmap.png
outputs/figures/eval_distill_loss_comparison.png
outputs/tables/evaluation_summary.csv
```

## 9. Blind Human Pairwise Evaluation

This optional evaluation compares baseline MoConGPT outputs against LoRA outputs
with model identities hidden from the evaluator. It does not train a model and
does not use multi-candidate preference fine-tuning.

Export held-out style prompts from the small test split:

```powershell
<env-python> scripts\export_eval_prompts.py `
  --prompt-csv data\style_subset_small\style_test_small.csv `
  --caption-column caption `
  --max-items 40 `
  --output outputs\human_pairwise_eval\test_prompts.txt
```

Generate baseline BVH outputs for the same prompts:

```powershell
<env-python> scripts\run_baseline_prompts.py `
  --prompt-file outputs\human_pairwise_eval\test_prompts.txt `
  --output-dir outputs\human_pairwise_eval\baseline_bvh `
  --python <env-python>
```

Generate LoRA BVH outputs from the continued adapter:

```powershell
<env-python> scripts\generate_with_style_lora.py `
  --checkpoint outputs\finetune_lora_200_cont_lr5e5\style_lora_last.pth `
  --prompt-file outputs\human_pairwise_eval\test_prompts.txt `
  --output-dir outputs\human_pairwise_eval\lora_bvh `
  --max-length 50 `
  --seed 7
```

Build the blind A/B review page:

```powershell
<env-python> scripts\build_blind_pairwise_eval.py `
  --prompt-file outputs\human_pairwise_eval\test_prompts.txt `
  --baseline-dir outputs\human_pairwise_eval\baseline_bvh `
  --lora-dir outputs\human_pairwise_eval\lora_bvh `
  --output-dir outputs\human_pairwise_eval `
  --seed 7
```

Open `outputs\human_pairwise_eval\review.html`, rate A/B for semantic match,
style match, and plausibility, choose A/B/Tie, export `preference_labels.csv`,
and place it in `outputs\human_pairwise_eval`.

Summarize the blind labels:

```powershell
<env-python> scripts\summarize_blind_pairwise_eval.py `
  --blind-key outputs\human_pairwise_eval\blind_key.csv `
  --labels outputs\human_pairwise_eval\preference_labels.csv `
  --output-dir outputs\human_pairwise_eval
```

Expected outputs:

```text
outputs/human_pairwise_eval/blind_key.csv
outputs/human_pairwise_eval/review.html
outputs/human_pairwise_eval/summary.csv
outputs/human_pairwise_eval/per_item_results.csv
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
