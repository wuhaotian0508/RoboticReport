# MoConVQ Baseline Reproduction Log

Date: 2026-05-21
Workspace: D:\roboticsreport\MoConVQ
Environment: .venv Python 3.8.20 managed by uv
GPU: CUDA available with torch 2.4.1+cu118

## Assets

- moconvq_base.data: present
- simple_motion_data.h5: present
- text_generation_GPT.pth: present
- unconditional_GPT.pth: present
- opensource.zip: retained as backup

## Environment Fixes

- Installed torch/torchvision/torchaudio cu118, transformers, and sentencepiece.
- Built RotationLibTorch from diff-quaternion/TorchRotation.
- Built VclSimuBackend from ModifyODESrc.
- Added single-process MPI fallback for systems without MS-MPI runtime.
- Used HF_ENDPOINT=https://hf-mirror.com and NO_PROXY=* for T5-large download.

## Reproduction Results

- Tokenization succeeded: out/tokens.txt
- Token decoding succeeded: out/decode.bvh
- Default text-to-motion prompt succeeded: out/conditional/evaluate_gpt0.bvh
- Project prompt batch succeeded: out/project_prompts/evaluate_gpt0.bvh ... evaluate_gpt9.bvh

See out/project_prompts/baseline_manifest.csv for prompt-to-file mapping.
