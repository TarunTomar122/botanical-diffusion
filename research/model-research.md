# Model Research — Small Diffusion From Scratch (5k–50k)

**Decision:** Pixel UNet DDPM++ 35–55M at 128px, from scratch, no SD fine-tune.

---

## TL;DR Recipe (for 3–5k Curtis plates, cheap GPU 12–24GB)

| Choice | Setting | Why |
|--------|---------|-----|
| Resolution | **128×128** (64 fallback if <3k heterogeneous) | 64 smudges 1px veins; 256 needs 4× data, will memorize |
| Architecture | **Pixel UNet DDPM++ (EDM `ddpmpp` / diffusers `UNet2DModel`)** | Proven for CIFAR 50k 35.7M, Flowers 8k, Pokemon 800, AFHQ 15k 64px |
| Size | **Small 35–55M**: `block_out_channels=(128,128,256,256,512,512)`, `layers_per_block=2`, attn at 16px | Tiny 5–12M underfits venation; 100M overfits + OOM |
| Scheduler | DDPM linear beta 1000 steps, DDIM/PNDM 50 infer | Standard |
| Batch | effective **32** (`per_device 8 × accum 4` on 12GB) | Stable EMA; 32 minimum |
| Steps | **400–800k** (~800–2500 epochs on 5k) | 100 epochs = demo, not converged; EDM = 200M images seen |
| Optimizer | AdamW `lr=1e-4` cosine, warmup 2000, `betas (0.95,0.999)`, `weight_decay 1e-6`, `EMA 0.9999`, dropout `0.1–0.15` | Small-data regularization |
| Augment | `Resize(128)→CenterCrop(128) + RandomHorizontalFlip 0.5` only | Preserve pigment hue, orientation; no vertical flip/color jitter |
| Precision | fp16/bf16 + xformers | Halves VRAM, 1.6× speed |

```python
UNet2DModel(
  sample_size=128, in_channels=3, out_channels=3,
  layers_per_block=2,
  block_out_channels=(128,128,256,256,512,512),
  down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
  up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"),
  dropout=0.1
)
```

Launch:
```bash
accelerate launch examples/unconditional_image_generation/train_unconditional.py \
  --train_data_dir ./data/botanical-128 \
  --resolution 128 --center_crop --random_flip \
  --output_dir runs/botanical-128-unet-s \
  --train_batch_size 8 --gradient_accumulation_steps 4 \
  --num_epochs 1200 --learning_rate 1e-4 --lr_warmup_steps 2000 \
  --lr_scheduler cosine --adam_beta1 0.95 --adam_beta2 0.999 --adam_weight_decay 1e-6 \
  --use_ema --ema_max_decay 0.9999 \
  --mixed_precision fp16 --enable_xformers_memory_efficient_attention \
  --dataloader_num_workers 4 --save_images_epochs 10 --save_model_epochs 20 \
  --ddpm_num_steps 1000 --ddpm_beta_schedule linear \
  --checkpointing_steps 10000
```

---

## Prior Work Table

| Work | Size | Res | Params | Result | Relevance |
|------|------|-----|--------|--------|-----------|
| DDPM Ho 2020 | CIFAR 50k | 32 | 35.7M UNet | FID 3.17 | Canonical small UNet |
| EDM Karras 2022 | CIFAR 50k, FFHQ 70k, AFHQ ~15k | 32/64 | 54–62M ddpmpp/ncsnpp, 113M FFHQ | CIFAR 1.97, ImageNet-64 1.36 | **Most important:** AFHQ 15k proves 15k works with dropout 0.25+augment 0.15 |
| ADM | ImageNet 1.2M | 64/128/256 | 296M | FID 2.97 @128 | 100M already strong, validates attn at 16–32px |
| HF diffusers examples | Flowers 8k, Pokemon 800, Butterflies 5.6k | 64/128 | 36–62M | Qualitative | **Proof 800 works**, 100 epochs demo |
| DiT Peebles 2022 | ImageNet 1.2M latent 32 | 256 latent | 33M–675M | 2.27 @256 XL/2 | Not for 5k — needs latent VAE + >300k |
| P2 Weighting Choi 2022 | — | — | — | +FID via mid-noise reweight | 1-line change, good for linework |

**Conclusion:** Field converged on **pixel UNet 30–60M at 64–128px** for 5k–50k. DiT/LDM win only with 1M+ or pretrained latents. UNet's translation equivariance is sample-efficient for radial flowers/leaf symmetry.

---

## Resolution Tradeoff

| Aspect | 64 | **128** | 256 |
|--------|----|---------|-----|
| Pixels | 4k | 16k (4×) | 65k (16×) |
| Detail | Coarse, loses veins | **Keeps 2px veins, serration, paper texture** | Publication lines |
| VRAM batch16 | ~7GB | ~10–12GB | >24GB OOM |
| Data need | 1–5k ok | **3–15k to avoid memorization** — 5k ok with reg | >20–50k |
| Time 200k steps | ~7h 3060 12GB | ~14h 3060, ~6h 4090 | 4× slower |
| FID stability | Stable | Borderline but ok | Noisy (<10k ref) |

**Why 128:** Botanical plates designed for 256+ dpi linework. 64 bilinear anti-aliases 1px lines → model learns blur priors. 128 preserves them; still fits cheap GPU.

---

## Small-Data Tricks

- **EMA 0.9999** critical — samples without are mottled
- **Dropout 0.1–0.3** inside ResBlocks (EDM AFHQ 0.25 for 15k)
- **x-flip 0.5** valid (plants bilaterally imperfect, engraving mirrored)
- **No** vertical flip, no strong color jitter (pigment encodes species), no cutout (destroys veins)
- **Preprocess:** Inpaint/remove text cartouches/scale bars/plate numbers before training or model hallucinates gibberish Linnaean text. Pad to square with `#F5F0E8` then resize (plates are portrait ~2:3).
- **Paper mode collapse:** 80% pixels are paper → model ignores plant. Keep paper constant via threshold fill or use v-prediction weighting.
- **P2 loss weighting** (upweight SNR -1 to +5) emphasizes structure over paper grain.

---

## VRAM Math

| Tier | Params | Example channels | 64 batch16 | 128 batch16 | 256 batch8 |
|------|--------|-----------------|------------|-------------|------------|
| Tiny | 5–12M | (64,128,128,256) | ~4GB | ~8GB | ~16GB OOM |
| **Small** | **30–55M** | (128,128,256,256,512,512) | ~7GB | **~10–12GB** | >24GB |
| Medium | 80–130M | (192,384,768) | ~12GB | ~20GB | OOM |

---

## Evaluation for 5k

- **FID-50k** vs entire training set (trend, not absolute; reference noisy)
- **KID** better for 5k (unbiased, low variance)
- **Precision/Recall, Density/Coverage** for memorization vs diversity
- **LPIPS nearest-neighbor** avg <0.2 = memorization
- **Human:** Double-blind real vs generated + botanist plausibility

---

## References

1. Ho et al. DDPM NeurIPS 2020 (2006.11239) — 35M CIFAR SOTA
2. Karras et al. EDM NeurIPS 2022 (2206.00364) + NVlabs/edm — best from-scratch recipe
3. Dhariwal & Nichol ADM 2105.05233 + openai/guided-diffusion
4. Peebles & Xie DiT 2212.09748 — why not transformer for 5k
5. Choi et al. P2 Weighting 2204.00227
6. HF diffusers unconditional training + train_unconditional.py
7. HF hub: `google/ddpm-cifar10-32`, `anton-l/ddpm-ema-flowers-64`, `huggan/pokemon`

Upgrade path: 128 proof → transfer to 256 → latent DiT-S/4 only if >30k → SD LoRA baseline for gap.
