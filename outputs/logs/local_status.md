# Local Status

Checked on 2026-05-21.

## Present

- MoConVQ source code exists under `MoConVQ/`.
- `base.bvh` and `track.bvh` example motions exist.
- LaTeX is installed (`pdfTeX 3.141592653-2.6-1.40.27`).
- Official MoConVQ pretrained assets are present in `MoConVQ/`.
- Baseline tokenization, decoding, default text-to-motion, and ten prompt BVH generation have been verified.

## Missing External Assets

- HumanML3D text annotations and official split files at `D:\roboticsreport\datasets\HumanML3D`.

## Consequence

Baseline reproduction is complete. Style subset statistics, fine-tuning, FID, and R-Precision are pending until HumanML3D and tokenized MoConVQ targets are prepared.
