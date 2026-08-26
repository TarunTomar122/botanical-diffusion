#!/usr/bin/env python3
import pathlib, random
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CREAM = (250,246,232)
raw = pathlib.Path("data/raw/curtis")
out = pathlib.Path("figures/dataset-inspection")
out.mkdir(parents=True, exist_ok=True)

# find jp2
files = list(raw.rglob("*.jp2"))
print(f"found {len(files)} jp2")
# sort by size descending
files_sorted = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
print("largest 10:")
for p in files_sorted[:10]:
    print(f"  {p.name} {p.stat().st_size//1024}KB")

# Take top 64 as likely plates, next 64 as medium, smallest 64 as text
top64 = files_sorted[:64]
mid64 = files_sorted[len(files_sorted)//2-32:len(files_sorted)//2+32]
small64 = files_sorted[-64:]

def make_grid(file_list, out_path, title, n=16):
    picks = file_list[:n]
    cols = 4
    rows = n//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < len(picks):
            try:
                with Image.open(picks[i]) as im:
                    # quick crop bottom 8% and pad to square, resize 256 for display
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    w,h = im.size
                    im = im.crop((0,0,w,int(h*0.92)))
                    w,h = im.size
                    max_side = max(w,h)
                    pad_w = max_side - w
                    pad_h = max_side - h
                    im = ImageOps.expand(im, (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2), fill=CREAM)
                    im = im.resize((256,256), Image.LANCZOS)
                    ax.imshow(im)
                    ax.set_title(f"{picks[i].name[:18]}\n{picks[i].stat().st_size//1024}KB", fontsize=6)
            except Exception as e:
                ax.text(0.5,0.5,str(e)[:30],ha='center',fontsize=6)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")

make_grid(top64, out / "grid_top64_likely_plates.png", "Top 64 largest (likely plates)", n=16)
make_grid(mid64, out / "grid_mid64.png", "Middle 64 (mixed)", n=16)
make_grid(small64, out / "grid_small64_text.png", "Smallest 64 (likely text/noise)", n=16)

# Also generate 8 samples for data/samples
samples_dir = pathlib.Path("data/samples")
samples_dir.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(top64[:8]):
    with Image.open(p) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w,h = im.size
        im = im.crop((0,0,w,int(h*0.92)))
        w,h = im.size
        max_side = max(w,h)
        pad_w = max_side - w
        pad_h = max_side - h
        im = ImageOps.expand(im, (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2), fill=CREAM)
        im.thumbnail((512,512), Image.LANCZOS)
        im.save(samples_dir / f"sample_{i:02d}.jpg", quality=92)
        print(f"sample {i} -> {p.name}")

# stats
sizes = [p.stat().st_size//1024 for p in files]
print(f"stats: count {len(sizes)} min {min(sizes)}KB max {max(sizes)}KB mean {np.mean(sizes):.0f}KB")
# histogram
plt.figure(figsize=(6,3))
plt.hist(sizes, bins=30, color='teal', alpha=0.7)
plt.title("JP2 filesize distribution (KB)")
plt.xlabel("KB")
plt.ylabel("count")
plt.axvline(500, color='red', linestyle='--', label='500KB thresh')
plt.legend()
plt.tight_layout()
plt.savefig(out / "filesize_hist.png", dpi=150)
plt.close()

# Write markdown
with open(out / "README.md","w") as f:
    f.write("# Dataset Inspection — Curtis's Botanical Magazine (sample 2 vols)\n\n")
    f.write(f"- **Files found:** {len(files)} jp2 (2 volumes s1id13292280, s1id13292270)\n")
    f.write(f"- **Size range:** {min(sizes)}–{max(sizes)} KB, mean {np.mean(sizes):.0f} KB\n")
    f.write(f"- **Heuristic:** largest files = plates (color, high detail). Threshold ~500KB separates plates (~33% >500KB) from text.\n\n")
    f.write("## Largest 16 (likely plates)\n\n")
    f.write("![top](grid_top64_likely_plates.png)\n\n")
    f.write("## Smallest 16 (likely text pages)\n\n")
    f.write("![small](grid_small64_text.png)\n\n")
    f.write("![hist](filesize_hist.png)\n")
print("done")
