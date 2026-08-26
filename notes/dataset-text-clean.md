# Text Contamination Fix (2026-08-26)

**Problem:** User noticed generated images contained text artifacts. Investigation found **5% (18/360)** of the 360-plate dataset were text-like pages slipping through the filesize-only filter (top40 per vol). They had low saturation (11 vs 26 for plates) and high edge density.

**Fix:** New `scripts/build_clean.py` — per vol, take top60 largest jp2, then filter by saturation >15 (64px thumbnail), keep 40. This removes text pages while keeping colorful plates.

**Local clean:** 360 → 342 plates (removed 18). `data/processed/curtis-128-clean` → now `data/processed/curtis-128`.

**GPU full dataset:** Restarting `data/raw/curtis_full` download (100 vols) with clean filter, targeting 100*40=4000 plates. Aiming for 5-10k total (supplement with BHL Flickr if needed). This will be built as `curtis-128-full` on the GPU pod.

**Training impact:** Current exp001 baseline was trained on 360 with 5% text, hence text artifacts. Next run (exp002) will use 4000 clean plates, resume from current checkpoint (~8-10k sweet spot), with stronger regularization (dropout 0.2).
