# exp004 Final — Oxford Flowers-102 Fresh Training

## Result

A fresh 35.7M pixel UNet was trained from scratch on **8,189 Oxford Flowers-102 images** at 128px. The dataset is flower-only by construction, avoiding the Curtis scan artifacts, text pages, rulers, calibration cards, and labels that contaminated the historical archive pipeline.

Training stopped at **30,302 steps** after the model produced consistent flower-like generations. Final observed loss was approximately **0.0036**, with zero logged training errors and stable 91-100% GPU utilization.

## Checkpoints and Samples

- Final local checkpoint: `experiments/exp004_flowers102/checkpoints/checkpoint-30000/` (410MB)
- Generated sample grids: `experiments/exp004_flowers102/`
- Final montage: `figures/exp004_latest_30k.png`
- Progression: `figures/progression_exp004.png`

CLIP checks at 1,250 and 5,250 steps classified all four latest tiles as flower/plant rather than text or calibration artifacts. At 30k, samples visibly developed varied flower-like structures. The model remains painterly/soft at 128px, but is no longer producing the text-heavy behavior inherited from the contaminated Curtis checkpoint.

## Decision

Stopped at 30k instead of spending the remaining GPU time because the generations had reached the experiment's useful qualitative stage and further training risked overfitting. The final RTX PRO 4000 pod was terminated and the account verified with no running botanical pod.

## Caveat

This final run uses flower photographs rather than historical botanical plates. It is a clean, fast flower-generation control experiment. The Curtis research and failure modes remain documented in `notes/dataset-text-clean.md` and the exp001/exp002 artifacts.
