# Baseline Outputs

Baseline reproduction has already been run with the official MoConVQ pretrained files.

Verified outputs:

- `project_prompts/baseline_manifest.csv`
- `project_prompts/evaluate_gpt0.bvh` through `project_prompts/evaluate_gpt9.bvh`

To regenerate them from Git Bash:

```bash
cd /d/roboticsreport
./MoConVQ/.venv/Scripts/python.exe scripts/run_baseline_prompts.py
```
