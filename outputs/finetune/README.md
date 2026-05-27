# Fine-Tuning Outputs

Run `scripts/train_style_adapter.py` after preparing HumanML3D style subsets and tokenized MoConVQ targets.

Expected outputs:

- `train_log.csv`
- `style_adapter_last.pth`
- `style_adapter_best.pth`

Current midterm status:

- `training_diagnostic.md` is expected before token targets exist.
- Loss curves and checkpoints must not be reported until the training loop has measured them.
