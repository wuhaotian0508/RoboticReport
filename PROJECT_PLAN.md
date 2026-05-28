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
   - LoRA train/validation pseudo-token loss evaluation
   - BVH motion-style proxy metric evaluation
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

- Diagnose HumanML3D/MoConVQ format compatibility.
- Use pretrained MoConVQ as a teacher to create pseudo token targets when true MoConVQ token labels are unavailable.
- Freeze the pretrained generator and train only LoRA updates using `scripts/train_style_lora_distill.py`.
- Save loss logs and checkpoints under `outputs/finetune_lora_200/`.
- Continue the selected LoRA checkpoint with a lower learning rate when measured proxy metrics improve without changing the evaluation rule.

Current status: completed for a 200-caption LoRA distillation run plus a two-epoch low-learning-rate continuation. The run trains 1,966,080 parameters out of 195,591,680 total parameters and writes `train_log.csv`, `style_lora_last.pth`, and epoch checkpoints.

### Stage 4: Evaluation

- Generate baseline and adapted motions for the same style prompts.
- Render selected side-by-side qualitative comparisons.
- Summarize LoRA loss and qualitative artifacts using `scripts/summarize_lora_results.py`.
- Evaluate checkpoint-level train/validation pseudo-token loss using `scripts/evaluate_lora_distill_loss.py`.
- Compare generated BVH files with kinematic proxy metrics using `scripts/compute_bvh_proxy_metrics.py`.
- Do not report FID or R-Precision until a compatible evaluator bridge is implemented.

Current status: baseline and LoRA BVHs exist for the same ten prompts, and the selected continued-LoRA run has deterministic BVHs for the five style prompts. Three updated baseline/LoRA video pairs, a qualitative frame grid, a loss curve, an epoch summary table, train/validation pseudo-token loss, and BVH style proxy metrics have been generated. The final selected LoRA run improves 18 of 22 predefined style-direction proxy comparisons, but does not improve every metric.

### Stage 5: Reporting

- Midterm report: honest progress, pipeline, data curation, preliminary or diagnostic results.
- Final report: full method, quantitative metrics if available, qualitative comparison, limitations, and backup variants.

## Current Local Workspace Status

This workspace contains MoConVQ source code, `base.bvh`, `track.bvh`, official pretrained assets, HumanML3D parquet data, verified baseline outputs, and a measured LoRA distillation run. The package avoids fabricated FID and R-Precision values because the available HumanML3D parquet features are not direct MoConVQ token supervision and no compatible official evaluator bridge has been verified. The reported quantitative evidence is limited to measured pseudo-token distillation loss and BVH kinematic proxy metrics.
