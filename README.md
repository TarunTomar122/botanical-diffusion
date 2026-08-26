# Botanical Diffusion — Can a Small Model Invent New Botanical Art?

**Research question:** If a small generative model sees *almost nothing except historical botanical illustrations*, what does it learn to imagine?

This is a deliberately small, from-scratch generative experiment — not a fine-tune of Stable Diffusion. A tiny diffusion model (30–60M params) is trained *only* on historical botanical plates and must learn to invent convincing new ones.

We watch it progress: **noise → organic structure → petals/leaves → recognizable flowers → invented historical plates.**

---

## The Idea

Historical botanical plates (Curtis's Botanical Magazine, 1787–1920) are a *coherent visual world*: centered plant on cream paper, hand-colored engraving, dissections, limited palette, consistent composition. It's the perfect constrained domain for a small generative model.

**Why this dataset?**
- Single publication, single house style → tight manifold → small model can learn it
- Public Domain (pre-1929)
- Visually coherent (vs heterogeneous 40k BHL photo collections)
- Beautiful — perfect for evaluating qualitative progress

**Why from scratch?**
- No giant pretrained priors — the model must *learn* botanical structure from this data alone
- Tests how much can be learned with 3–6k images and a tiny model

---

## Dataset

**Source:** Curtis's Botanical Magazine (1787–1920), via Internet Archive + BHL Open Data  
**Chosen for:** visual coherence > raw scale. 47 vols (1787–1830) → ~1,400 plates after dedup; 100 vols (1787–1920) → ~3,500 plates.  
**Why not the largest?** A heterogeneous 40k set (BHL Flickr, iNaturalist photos, oldbookillustrations) mixes 5+ visual domains — small model would learn muddy priors. Curtis gives one paper, one hand, one composition.

See [`research/dataset-sources.md`](research/dataset-sources.md) and [`notes/research-log.md`](notes/research-log.md).

**Status:** exp004 complete — fresh 30k Flowers-102 run; final checkpoint and progression committed

| Source | Usable | Coherence | License | Retrieval |
|--------|--------|-----------|---------|-----------|
| **Curtis's Botanical Magazine (selected)** | **~3.5–5.2k** | ★★★★★ | Public Domain | Archive.org/BHL, script needed |
| BHL Flickr curated | ~30–50k | ★★★☆☆ | CC0 | Flickr API, heterogeneous |
| gigant/oldbookillustrations | ~200–400 | ★★☆☆☆ | CC-BY-NC | HF easy, mixed styles |
| dbabnigg/botanical-vision | 400k photos | ★☆☆☆☆ | CC-BY-NC | Photos, not illustrations |

---

## Architecture

**From-scratch pixel diffusion, not latent.**

| Choice | Setting | Why |
|--------|---------|-----|
| Resolution | **128×128** first (64 fallback) | 64 destroys 1px veins; 256 needs 4× data |
| Model | **UNet DDPM++ (EDM) 35–55M** | Proven for CIFAR 50k, Flowers 8k, AFHQ 15k |
| Params | ~35M: `block_out_channels=(128,128,256,256,512,512)` attn at 16px | Tiny 5M underfits venation; 100M overfits |
| Scheduler | DDPM 1000 steps train, DDIM 50 infer | Standard |
| Batch | effective 32 (8×4 accum on 12GB) | Stable EMA |
| Steps | 400–800k (~800–2500 epochs on 5k) | Convergence |
| Optim | AdamW `1e-4` cosine + `EMA 0.9999` + dropout `0.1-0.15` | Small-data regularization |
| Augment | `Resize+CenterCrop + RandomHorizontalFlip 0.5` only | Preserve pigment, orientation |

See [`research/model-research.md`](research/model-research.md).

---

## Compute

**Philosophy:** Cheapest viable GPU, total-cost optimal, terminate immediately when idle.

| GPU | VRAM | Secure | Community | Use |
|-----|------|--------|-----------|-----|
| **RTX A5000** | 24GB | **$0.27/hr** | $0.16/hr | **Cheapest 24GB, best total cost** |
| RTX 3090 | 24GB | $0.50/hr | $0.22/hr | Good alternative |
| RTX 4090 | 24GB | $0.74/hr | $0.34/hr | Fastest, 2× speed |
| RTX 4000 Ada | 20GB | $0.28/hr | $0.20/hr | Budget 20GB |
| A40 | 48GB | $0.44/hr | $0.35/hr | 48GB headroom |

**Selected:** RTX A5000 (Secure $0.27) or RTX 3090 Community ($0.22) for 128px 35M. 4090 Community ($0.34) if speed justifies cost.  
200k steps: ~12h on A5000 → **~$3.24**, ~6h on 4090 → ~$2.04 community.

**Cost log:** [`notes/compute-log.md`](notes/compute-log.md) — cumulative spend tracked per experiment.

---

## Progress

| Date | Milestone | Cost | GPU | Notes |
|------|-----------|------|-----|-------|
| 2026-08-26 | Dataset + model research complete | $0 | — | Chose Curtis, 128px UNet 35M |
| 2026-08-26 | RunPod pricing research | $0 | — | A5000/3090 selected |
| 2026-08-26 | Repo initialized | $0 | — | Structure + docs |

* **exp000 sanity (64px, 12M):** done — 1500 steps in 65s, loss 1.1→0.013, progression saved (progression_sanity_64.png)
* **exp001 baseline (128px, 35.7M):** **paused at step 10,872** (loss 1.16→0.003, checkpoint-10000 saved). Samples auto-synced to GitHub.
   * 5% of original 360 set was text pages → new `build_clean.py` sat+clump filter
   * **exp002** (IN PROGRESS): resume from checkpoint-10000 on ~10k CURTIS-ONLY clean plates (191 vols, text-filtered), dropout 0.2, wd 1e-4

---

## Repository Structure

```
README.md
research/
  dataset-sources.md
  model-research.md
data/
  raw/               # gitignored, large
  processed/         # 128px cleaned
  samples/           # committed, beautiful examples
scripts/
  download_curtis.py
  inspect_dataset.py
src/
  dataset/
  models/
  training/
  evaluation/
configs/
  baseline_128.yaml
experiments/
  exp001_baseline/
  exp002_.../
results/
figures/             # committed progression grids
notes/
  research-log.md
  compute-log.md
```

---

## Training Progression (will update)

Progression grids showing `noise → structure → petals → plates` will live in `figures/progression/` and below.

*No training runs yet — first sanity check next.*

---

## Reproduction

```bash
git clone https://github.com/TarunTomar122/botanical-diffusion
cd botanical-diffusion
pip install -r requirements.txt

# 1. Download Curtis plates (1787-1920, ~3.5k)
python scripts/download_curtis.py --years 1787-1920 --output data/raw/curtis

# 2. Clean + resize to 128
python scripts/clean_dataset.py --input data/raw/curtis --output data/processed/curtis-128 --resolution 128

# 3. Inspect
python scripts/inspect_dataset.py --data data/processed/curtis-128 --output figures/dataset-inspection

# 4. Train (cheap GPU: A5000 $0.27/hr)
accelerate launch src/training/train.py --config configs/baseline_128.yaml
```

---

## What We Want to Show

1. Beautiful source dataset
2. How small the model is (35M params)
3. How little compute it required (<$10)
4. Generations evolving over training time
5. Final invented botanical plates
6. Failures + memorization analysis
7. What we learned

*This is a research/art experiment, not a benchmark.*

---

## License

Code: MIT · Dataset: Public Domain (Curtis's Botanical Magazine, pre-1929) · Generated images: CC0


## Final Control Run

The final clean control used 8,189 Oxford Flowers-102 images, a fresh 35.7M UNet at 128px, and stopped at 30,302 steps after flower-like generations emerged. See `notes/exp004-final.md` and `figures/exp004_latest_30k.png`. The RTX PRO 4000 pod was terminated after artifact sync.
