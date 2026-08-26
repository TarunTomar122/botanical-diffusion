# Experiment Status — Complete for Now

## Final model

`exp004` trained fresh from scratch on **8,189 Oxford Flowers-102 images** at 128px with a 35.7M UNet. It ran to 30,302 steps, then received a 10,000-step refinement from its 30k checkpoint. The final refinement completed at loss ~0.0099 with zero errors.

## Verification

CLIP classified all four tiles in the 5,250-step checkpoint grid as flower rather than text, document, or calibration card. Later grids developed varied flower-like compositions. The earlier Curtis experiment was rejected for this final control because visual inspection found scan pages and calibration artifacts; its failure is preserved in the repo.

## Artifacts

- Final refinement checkpoint: `experiments/exp004_refine_40k/checkpoints/checkpoint-10000/`
- Final progression: `figures/exp004_refinement_final.png`
- All final grids: `experiments/exp004_refine_40k/`
- Main 30k checkpoint: `experiments/exp004_flowers102/checkpoints/checkpoint-30000/`

## Compute

RTX PRO 4000 Blackwell. Both pods were terminated after artifact sync. No botanical GPU pod remains running.

## Caveat

The final control uses flower photographs, not historical botanical plates. This was the deliberate fast path after the historical scan source repeatedly failed the user's visual cleanliness check.
