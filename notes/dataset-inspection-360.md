# Dataset Inspection — 360 plates (9 vols)

**Date:** 2026-08-26 06:56
**Volumes:** 9 (s1id13292120,30,40,50,60,70,80,270,280) — 1432 jp2 total
**Processed:** 360 plates at 128 and 360 at 64 (top 40 per volume by filesize)

**Method:** Per volume, sort jp2 by filesize descending, take top 40. 500KB thresh ~33% plates, 600KB 12% — top40 balances. Crop bottom 8% caption, pad to square cream #faf6e8, LANCZOS resize.

**Counts:** 9 vols × 40 = 360. Each image ~6-9KB at 128, ~2-3KB at 64.

**Grids:**
- `grid_processed_128_360.png` — random 16 from 360 (shown below)
- Previous 2-vol grids still valid.

**Quality:** All plates show centered plant, cream paper, dissections, consistent 1787-1830 style. No text pages in top40 — verified via filesize hist and visual.

**Next:** Use 360 for main 128 training (400k steps). For sanity, use 64 with same 360 or 40 subset.

**Storage:** `data/processed/curtis-128` 360×128, `data/processed/curtis-64` 360×64, ~2.8MB + ~1MB.
