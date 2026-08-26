# Compute Log — Botanical Diffusion

Tracks every paid GPU minute. Goal: small hobby experiment, minimize spend.

---

## Pricing Research (2026-08-26)

Queried via `api.runpod.io/graphql` with API key.

| GPU | VRAM | Secure $/hr | Community $/hr | Notes |
|-----|------|-------------|----------------|-------|
| RTX A4000 | 16GB | $0.25 | $0.17 | Tight for 128px, may OOM batch 16 |
| RTX A5000 | 24GB | **$0.27** | **$0.16** | **Cheapest 24GB — selected** |
| RTX 4000 Ada | 20GB | $0.28 | $0.20 | Budget 20GB |
| A40 | 48GB | $0.44 | $0.35 | 48GB headroom mid-cost |
| RTX 3090 | 24GB | $0.50 | $0.22 | Good alternative 24GB |
| RTX 4090 | 24GB | $0.74 | $0.34 | Fastest ~2× A5000 |
| L4 | 24GB | $0.49 | $0.44 | — |
| A100 80GB | 80GB | $1.39–1.59 | $1.19–1.39 | Overkill for 35M 128px |
| H100 80GB | 80GB | $2.89–3.29 | $1.99–2.69 | 10× cost, not justified |
| B200/H200 | 180GB | $4.59–6.79 | $3.59–5.98 | Not for this |

**Selection rationale:** 35M @128px batch 16 needs ~10–12GB. 24GB gives comfortable headroom with fp16+xformers+EMA. RTX A5000 Secure $0.27 is cheapest secure 24GB; Community $0.16 is cheapest total. RTX 4090 Community $0.34 is faster (400k steps 6h vs 12h) — total cost $2.04 vs $3.24, so 4090 community can be *cheaper total* despite higher hourly. Choose A5000/3090 for safety, 4090 if iteration speed matters.

**Time estimates (35M @128, batch effective 32, ~35 img/sec on A5000/3090, ~70 img/sec on 4090):**
- 50k steps (sanity): ~1.5h A5000 ($0.41) / 0.7h 4090 ($0.24 community)
- 200k steps: ~6h A5000 ($1.62) / 3h 4090 ($1.02 community)
- 400k steps: ~12h A5000 ($3.24) / 6h 4090 ($2.04 community)
- 800k steps: ~24h A5000 ($6.48) / 12h 4090 ($4.08 community)

---

## Runs

| # | Date | Experiment | GPU | Type | $/hr | Intended | Actual | Cost | Outcome | Continue? |
|---|------|------------|-----|------|------|----------|--------|------|---------|-----------|
| — | 2026-08-26 | Pricing query only | — | — | — | — | — | $0 | — | — |
| | | | | | | | | | | |

**Cumulative spend: $0.00**

---

## Existing Resource (pre-project)

- Pod `5fg3uvvzwsgf44` `attentionisallineed-pro4000` EXITED, 1 GPU, 31GB RAM, $0.57/hr, `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` — not used for botanical, left EXITED to avoid cost.

---

## Cost Discipline Rules

- Create pod only when actually training/evaluating
- Prepare code/data before GPU time
- Checkpoint every 5–10k steps
- Stop early if diverging/memorized/broken
- Terminate immediately when idle (research, docs, analysis done locally)
- Favor short exploratory → inspect → targeted improvement → larger only if justified
- No H100/A100 without clear evidence.

Before each run record: GPU, $/hr, hypothesis, duration, est cost. After: actual, cost, outcome, next.
## Updates 2026-08-26 06:15-06:40

**Pricing verified:** same table.

**Runs attempted:**
| # | Time | Experiment | GPU | Type | $/hr | Intended | Actual | Cost est | Outcome | Continue? |
|---|------|------------|-----|------|------|----------|--------|----------|---------|-----------|
| 1 | 05:51 | pod xno2q0zbalxrd6 sanity 3090 | RTX 3090 | Community | $0.22 | 15min sanity | 2min, terminated | $0.01 | 502/SSH refused after 2min, terminated early (premature) |
| 2 | 05:53 | pod ur5tkyazkb6yz2 3090 secure | RTX 3090 | Secure | $0.50 | 15min | 5min, terminated | $0.04 | 502/SSH refused after 5min, terminated |
| 3 | 05:58 | pod df8pl05dgmsmud L4 secure | L4 | Secure | $0.49 | 15min | 6min, terminated | $0.05 | 502/SSH refused after 6min, image pull slow |
| 4 | 06:03 | pod tnp1kvhdid25nv 4000 Ada secure | RTX 4000 Ada | Secure | $0.28 | 15min | 6min, terminated | $0.03 | 502/SSH refused after 6min |
| 5 | 06:09 | pod k7cn3s0nib0qs3 2000 Ada secure | RTX 2000 Ada | Secure | $0.24 | 15min | 6min, terminated | $0.02 | 502/SSH refused after 6min |
| 6 | 06:15 | pod uk0cul437cy96f ubuntu test | RTX 2000 Ada | Secure | $0.24 | test ubuntu 10GB | 2min, terminated | $0.01 | still provisioning after 80s, terminated |
| 7 | 06:35 | pod 2kw198f6efwr26 3090 30GB | RTX 3090 | Secure | $0.50 | sanity | running, uptime 374s at 06:41, still 502/SSH refused | $0.05 so far | **still running, polling** — will wait 10min total before verdict |

**Lesson:** Public IP appears in ~15s, but Jupyter/SSH needs 5-10 min for 20GB pytorch image pull. Early terminations after 3-6 min were premature — now waiting 10 min for 2kw. Next time use larger containerDisk 30GB and be patient. Also need to set `torch.set_num_threads(1)` for CPU host due to contention.

**Cumulative provisioning overhead:** ~$0.20 (no training spend yet).

**Next:** If 2kw still fails after 10 min, try different host/GPU (A40, RTX 6000 Ada) or use `runpodctl` or try community with smaller image.
