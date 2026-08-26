# Research Log — Botanical Diffusion

Chronological record of decisions, discoveries, failures, observations.

---

## 2026-08-26 — Project Initialization

**Goal defined:** Can a small generative model trained from scratch only on historical botanical illustrations invent convincing new botanical art? Watch progression noise → structure → petals → plates.

**Autonomy:** Full end-to-end, minimal human intervention. Responsible for research, data, model, compute, evaluation, GitHub, cost control.

---

### Dataset Research

**Investigated:**
- Curtis's Botanical Magazine (1787–present) via Archive.org/BHL — 100 docs, 24k imagecount, ~36 plates/vol
- BHL general (64M pages, 300k Flickr curated) — heterogeneous, low coherence raw
- HF datasets: `botanical` (6 hits, all modern photos), `curtis` (0 relevant), `biodiversity` (text-only), `gigant/oldbookillustrations` (4.1k heterogeneous, ~5% botanical, CC-BY-NC), `YakirLantern` (160 modern photos, broken), `dbabnigg/botanical-vision` (400k modern photos, wrong domain), `finebooks/bhl-impact-gt` (2k pages, not botanical)

**No existing HF dataset satisfies brief.** All historic botanical datasets are unmirrored to HF; Archive.org/BHL is source of truth.

**Decision: Curtis's Botanical Magazine 1787–1920 as primary corpus.**
- **Why:** Single serial, single house style, centered plant on cream, hand-colored engraving → lithograph, maximum visual coherence. Tight manifold ideal for small model.
- **Yield:** 47 vols (1787–1820) → ~1,400 after dedup; 100 vols → ~3,500; 150 vols → ~5,200. 5k is sweet spot for small diffusion.
- **Coherence vs scale:** Flickr 40k heterogeneous mixes Thornton/Redouté/Besler/Danica — 3× style variance forces small model to learn marginal distribution, not botanical structure. Curtis wins.
- **Licensing:** Public Domain (pre-1929). No NC restriction.
- **Retrieval:** Archive.org `s1id13292*` + `mobot31753002*` IDs, `*_jp2.zip` downloads, or BHL S3 `bhl-open-data/images/[BarCode]/`. Requires script + plate-page filtering (scandata.xml `pageType==Illustration`, color histogram, phash dedup, auto-crop bottom 8% caption).
- **Fallback:** If <3k after dedup, expand to 1920 or supplement with BHL Flickr curated `botany` tag (but log style heterogeneity).

**Recorded in:** `research/dataset-sources.md` (full table to be added)

---

### Generative Model Research

**Investigated:**
- DDPM (Ho 2020) — 35.7M UNet, CIFAR-10 32px, FID 3.17
- EDM (Karras 2022) — 54–62M ddpmpp/ncsnpp, FFHQ 70k + AFHQ ~15k at 64px, dropout 0.25, augment 0.15 — most relevant. Training 200M images seen.
- ADM (Dhariwal 2021) — 296M, ImageNet 128, not for 5k
- HF diffusers unconditional examples — `train_unconditional.py`, Flowers 8k, Pokemon 800, Butterflies 5.6k, CelebA — direct proof 800 works with 36–62M
- DiT (Peebles 2022) — Transformer needs >300k ImageNet, latent VAE, not for 5k. UNet inductive bias wins.
- P2 weighting, U-ViT, Latent Diffusion

**Decision: Pixel-space UNet DDPM++ 35–55M at 128px.**
- **Why 128 not 64?** 64 anti-aliases 1px veins into gray smudges; 128 preserves 2px veins/petal serration. 256 quadruples VRAM + needs 4× data, will memorize on 5k.
- **Architecture:** `UNet2DModel(sample_size=128, block_out_channels=(128,128,256,256,512,512), layers_per_block=2, attention at 16px)` ~35M. Dropout 0.1–0.15, EMA 0.9999.
- **Schedule:** DDPM 1000 steps, DDIM 50 infer, batch effective 32 (8×4 accum), AdamW 1e-4 cosine warmup 2000, 400–800k steps (800–2500 epochs on 5k).
- **Augment:** Resize+CenterCrop + RandomHorizontalFlip 0.5 only. No color jitter (pigment is signal), no vertical flip. Light affine ±3° optional phase 2.
- **Next upgrade:** If 128 converges, transfer to 256; else latent DiT only if >30k data.

**Recorded in:** `research/model-research.md` (to be added)

---

### Compute Research (RunPod)

**Queried:** GraphQL `gpuTypes { securePrice communityPrice }` via API key.

**Pricing sorted (Secure / Community):**
- RTX A5000 24GB: $0.27 / $0.16 ← **cheapest 24GB**
- RTX 4000 Ada 20GB: $0.28 / $0.20
- A40 48GB: $0.44 / $0.35
- RTX 3090 24GB: $0.50 / $0.22
- RTX 4090 24GB: $0.74 / $0.34 ← fastest
- L4 24GB: $0.49 / $0.44
- RTX 3080 Ti 12GB: — / $0.18 (too small for 128 batch 16)
- RTX 3070 8GB: — / $0.13 (OOM)
- A100 80GB: $1.39–1.59 / $1.19–1.39 (overkill)
- H100 80GB: $2.89–3.29 (overkill, 10× cost)

**VRAM need:** 35M @128 batch 16 → ~10–12GB. So 16GB minimum, 20–24GB comfortable with xformers fp16.

**Decision:**
- **Primary:** RTX A5000 Secure $0.27/hr (cheapest viable 24GB, reliable) or Community $0.16/hr (cheapest total)
- **Alternative:** RTX 3090 Community $0.22/hr (good balance) or RTX 4090 Community $0.34/hr (2× faster, total cost ~$2.04 for 6h 400k steps vs $3.24 for A5000 12h)
- **Philosophy:** Prefer A5000/3090; only use 4090 if clear economic reason (faster iteration). Optimize total cost, not hourly. Always checkpoint frequently for Community spot interruption.
- **Estimates:** 64px 35M batch32 → 120 img/sec on 3060 → 400k steps 7h; 128px batch32 effective → 35 img/sec → 400k steps 12h A5000 / 6h 4090.

**Recorded in:** `notes/compute-log.md`

**Current pod:** `attentionisallineed-pro4000` EXITED ($0.57/hr), not botanical-related. Will create new cheap pod when needed, terminate immediately after.

---

## Next Steps

1. Write `research/dataset-sources.md` + `research/model-research.md` summaries
2. Initialize `notes/compute-log.md` with pricing table
3. Build `scripts/download_curtis.py` for Archive.org → plate extraction
4. Download sample (1–2 vols) → inspect grids, stats, dedup
5. Implement training (`src/models/unet.py`, `src/training/train.py`) based on diffusers `train_unconditional.py` but locked to 35M 128px
6. Sanity check: tiny run (64px, 1k images, 5k steps) locally or cheap pod to verify pipeline before main run

---

## 2026-08-26 08:40-09:00 — GPU RUNS FINALLY WORK (RTX PRO 4000 Blackwell)

**Failure root cause discovered (with user's help):**
1. **Missing PUBLIC_KEY:** RunPod pods only start `sshd` when `env.PUBLIC_KEY` is set at creation. All my earlier pods lacked it → `Connection refused`. Old working pod had `ttomar@adobe.com` key.
2. **Broken hosts:** one pod (`5ufj5rde3dcouw`) returned host Docker error `error creating overlay mount ... no such file or directory` — container never started, `runtime=null` forever.
3. **Fix:** `env: [{key:"PUBLIC_KEY", value:"<local ed25519 pubkey>\n<local rsa pubkey>"}]` + user-picked **RTX PRO 4000 Blackwell** (24GB, $0.57).

**Sanity (exp000):** 64px 12M, 1500 steps, batch16, EMA, no amp → **65s total, 24 it/s, loss 1.1→0.013**, 6 sample grids saved+committed, montage `figures/progression_sanity_64.png`.

**Baseline (exp001):** 128px 35.7M, batch 8, EMA, cosine→50k steps. **LIVE at 3.5 it/s**, loss 1.16→0.03 by step 300 and ~0.05-0.06 at 400. ETA ~4h → est $2.28 on PRO4000. Checkpoints 2k, samples 1k. Auto-poller syncing samples+montage to GitHub every 5min.

**Next:** let exp001 run; poll; at 10k inspect grids qualitatively; at ~50k evaluate: montage, memorization NN check, loss curve; then decide on extension (more steps / higher res / flowers-only).
