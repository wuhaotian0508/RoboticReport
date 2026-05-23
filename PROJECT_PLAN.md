# Project Plan

## Topic

Style-aware adaptation for language-conditioned humanoid motion generation based on MoConVQ.

The project follows the course direction "Language-guided motion generation": reproduce a pretrained MoConVQ text-to-motion pipeline, curate a style-focused HumanML3D subset, adapt only the language-conditioning transformer, and evaluate style fidelity without retraining the motion representation backbone.

## Hard Constraints

- Do not change the topic after the accepted proposal.
- Do not train MoConVQ representation models from scratch.
- Do not fabricate FID, R-Precision, training loss, or qualitative observations.
- Preserve the official HumanML3D train/validation/test split. Filtering must be applied independently inside each split.
- Emphasize humanoid robot motion generation and physics-based motion control, not only animation.

## Work Packages

### Batch 1: Midterm Deliverables

1. RSS-style midterm report, 1--4 pages.
2. Method pipeline figure.
3. Style subset construction script and statistics table.
4. Baseline prompt list and baseline generation script.
5. Fine-tuning implementation or, if training cannot be run, a technical diagnostic table.
6. References in BibTeX format.

### Batch 2: Final Deliverables

1. Final report with method, implementation, experiments, results, limitations, and future work.
2. Reproducible code entry points:
   - style subset filtering
   - small split creation
   - baseline generation
   - style-adapter fine-tuning
   - FID/R-Precision evaluation from extracted features
3. Prompt lists, audit template, and output folder structure.
4. Checklists for running the project once HumanML3D and pretrained weights are available.

## Execution Plan

### Stage 1: Environment and Baseline

- Install MoConVQ dependencies using `MoConVQ/setup.cmd`.
- Download official pretrained files into `MoConVQ/`.
- Verify required files:
  - `MoConVQ/moconvq_base.data`
  - `MoConVQ/text_generation_GPT.pth`
- Run at least ten baseline prompts and save BVH files under `outputs/baseline/`.

Current status: completed locally. The official pretrained files are present, tokenization/decoding works, and ten prompt BVH files exist under `MoConVQ/out/project_prompts/`.

### Stage 2: Data Curation

- Prepare HumanML3D split files and text annotations.
- Run `scripts/filter_style_subset.py`.
- Create small splits using `scripts/make_small_splits.py`.
- Fill `data/manual_audit_template.csv` with a 30-sample manual audit.

### Stage 3: Adaptation

- Tokenize selected HumanML3D motions using the MoConVQ tokenizer.
- Fine-tune only the text-to-motion transformer using `scripts/train_style_adapter.py`.
- Use fp16, gradient accumulation, and early stopping for RTX 4060 8 GB.
- Save loss logs and checkpoints under `outputs/finetune/`.

### Stage 4: Evaluation

- Generate baseline and adapted motions for the same style prompts.
- Extract motion features using the selected evaluator.
- Run `scripts/evaluate_motion_metrics.py` for FID and retrieval metrics.
- Report full-test retention and style-subset improvement.

### Stage 5: Reporting

- Midterm report: honest progress, pipeline, data curation, preliminary or diagnostic results.
- Final report: full method, quantitative metrics if available, qualitative comparison, limitations, and backup variants.

## Current Local Workspace Status

This workspace contains MoConVQ source code, `base.bvh`, `track.bvh`, official pretrained assets, and verified baseline outputs. It does not currently contain the external HumanML3D dataset. Therefore, this package avoids fabricated FID, R-Precision, or fine-tuning numbers and provides reproducible scripts plus report drafts that clearly mark the current diagnostic status.
