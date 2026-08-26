# Dataset Sources — Historical Botanical Illustrations

**Decision:** Curtis's Botanical Magazine (1787–1920) is the primary corpus. Visual coherence > raw scale.

---

## Why Coherence Matters

Small diffusion (35M) memorizes the *marginal style*. A heterogeneous 40k set (BHL Flickr mixing Thornton, Redouté, Besler, Flora Danica + modern iNaturalist photos) injects 5+ visual domains → muddy vintage, mode collapse, blurred paper. Curtis gives **one serial, one house style, one paper, one palette** — tight manifold the model can actually learn with 3–6k images.

---

## Candidates Investigated (2026-08-26)

| # | Candidate | Usable | Res | Coherence | Period/Style | Isolated? | Duplicates | License | Ease | Verdict |
|---|-----------|--------|-----|-----------|--------------|-----------|------------|---------|------|---------|
| **1** | **Curtis's Botanical Magazine (Archive.org/BHL)** | **3.5–5.2k** (47 vols→1.4k, 100 vols→3.5k) | 300 DPI JP2, 1600px deriv | ★★★★★ single serial, centered habit + dissections, cream paper | 1787–1920 hand-colored copper → lithograph | Mostly isolated, needs auto-crop caption (bottom 8%) | High if bulk (re-scans), phash dedup needed | **Public Domain** | Medium (script + filtering) | **SELECTED** |
| 2 | BHL Flickr curated | ~30–50k botanical of 300k | 1600px JPEG, human-cropped | ★★★☆☆ multi-source (Curtis, Thornton, Danica) | 1700–1900 mixed engraving/lith | ★★★★★ isolated by design | Low (curated) | CC0/CC-BY | Medium (Flickr API, no bulk zip) | Fallback expansion |
| 3 | gigant/oldbookillustrations | ~200–400 botanical of 4.1k | 1600px dual columns | ★★☆☆☆ 544 books, 483 artists, mixed subjects | 1800–1900 heterogeneous | Varies, borders/vignettes | Low | CC-BY-NC-4.0 | ★★★★★ HF `load_dataset` | Reject (too mixed, NC) |
| 4 | YakirLantern/Botanical | 160 | broken viewer, Hebrew names | ★☆☆☆☆ modern crop photos, not illustrations | 2024 photos | Isolated photos | — | Unknown | `load_dataset` fails | Reject |
| 5 | finebooks/bhl-impact-gt | 0 botanical of 2k | WebP page scans 1.8–3.7k | ★☆☆☆☆ 6 unrelated natural-history books | 1708–1913 birds/shells/fish | Heavy text/borders | — | CC-BY-3.0 | Easy HF | Reject |
| 6 | dbabnigg/botanical-vision | 408k | 256px photos | ★☆☆☆☆ modern iNaturalist RESEARCH-GRADE photos | 2020s photos | Field clutter | Deduped | CC-BY-NC | Easy HF | Reject (photos) |
| 7 | common-pile/bhl | 0 images (45M text rows) | N/A text only | N/A | 15th–21st c text | N/A | — | PD | Text only, images separate JP2 | Reject |
| 8 | Smithsonian Open Access | 2–5k herbarium sheets via IDS | 512px→high-res IDS | Low — specimen photos, not vintage | Mixed herbarium | Labels/rulers/barcodes heavy noise | — | CC0 | API, not tabulated | Reject (photo texture) |

**HF searches performed:** `botanical` (6 hits, all modern), `curtis` (4 hits, 0 botanical), `biodiversity` (46 hits, text-only), `historical botanical` (0), `plant illustration` (0). Archive.org `advancedsearch q=title:"Curtis's botanical magazine"` → 100 docs, 24k imagecount.

---

## Curtis Deep Dive

- **Archive.org IDs:** `s1id13292120–s1id13292560` (47 vols 1795–1820, 10.5k imagecount), `mobot31753002719505...` (7 vols), `curtissbotanica1481unse` (520pp 1845), `curtissflowergar0000unse` (120 plates 1787–1807 reprint, 258pp — easiest quick start).
- **Files:** `*_jp2.zip`, `*_pdf`, `scandata.xml`, `bhlmets.xml`. No pre-cropped plates zip.
- **Imagecount overcounts** by 3–5× (includes half-titles, indexes). Real plates ~15–30% of imagecount. Example `s1id13292280` (188 imagecount → ~35 plates).
- **BHL S3:** `s3://bhl-open-data/images/[BarCode]/[BarCode]_####.jp2` (300 DPI JP2), `s3://bhl-open-data/scandata/[BarCode]_scandata.xml` (`pageType=Illustration`), `data/item.txt.gz` maps BarCode→TitleID 410, `data/page.txt.gz` gives PageID/PageType. API v3 `api3?op=GetTitleMetadata&titleid=410&apikey=...`.
- **Plate extraction:** Filter `scandata.xml` `pageType==Illustration` + `addToAccessFormats`, color histogram (plates high saturation vs b/w text), auto-crop foot caption (bottom 8% always `Tab. 123 / Pub. by Curtis ...`), white-margin bbox, `phash` dedup across `mobot*` vs `s1id*` re-scans.
- **Normalize:** Resize shortest edge 512→128, keep aspect, pad to square with paper cream `#faf6e8`, save WebP.
- **Licensing:** Unambiguously Public Domain (1787–1928+ pre-cutoff, BHL marks `license: Public Domain`).
- **Metadata:** Rich: `BarCode`, `ItemID`, `PageID`, `SequenceOrder`, `PageType`, plate number, binomial OCR.

**Expected yields:**
- 47 vols (1787–1830 copper engraving, max coherence) → ~1,400 deduped
- 100 vols (1787–1920) → ~3,500 (recommended)
- 150 vols (→1950) → ~5,200 but post-1920 photography breaks style

**Retrieval commands:**
```bash
curl -s "https://archive.org/advancedsearch.php?q=title:%22Curtis%27s+botanical+magazine%22&fl=identifier,year,imagecount&rows=100&output=json" | jq
curl -L https://archive.org/download/curtissflowergar0000unse/curtissflowergar0000unse_jp2.zip -o curtis120.zip
aws s3 ls s3://bhl-open-data/images/ --no-sign-request | grep BarCode
```

---

## Why Not the Largest?

Small from-scratch diffusion learns *paper texture vs photo bokeh vs engraved border* if given heterogeneous data. 5k coherent teaches `petal venation, leaf phyllotaxy, watercolor wash`. Scaling heterogeneous to 40k adds little for vintage aesthetic, increases mem risk. Coherence lets tiny model succeed.

---

## Next: Build Pipeline

`scripts/download_curtis.py` will enumerate 100 Curtis IDs → download `_jp2.zip` → parse `scandata.xml` → keep Illustration pages → color filter → crop → dedup → resize 128 → manifest + inspection grids.

If <3k after dedup, optionally supplement with BHL Flickr `botany` tag but document heterogeneity.
