#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
CREAM=(250,246,232)
raw=Path("data/raw/curtis")
out=Path("figures/dataset-inspection")
out.mkdir(parents=True, exist_ok=True)
files = sorted(raw.rglob("*.jp2"), key=lambda p: p.stat().st_size, reverse=True)
print(f"total {len(files)}")
top16 = files[:16]
small16 = files[-16:]

def grid(file_list, out_path, title):
    cols=4; rows=4
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2.2))
    axes=np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < len(file_list):
            p=file_list[i]
            try:
                with Image.open(p) as im:
                    print(f"[{i}] opening {p.name} {p.stat().st_size//1024}KB {im.size}")
                    if im.mode!="RGB": im=im.convert("RGB")
                    w,h=im.size
                    im=im.crop((0,0,w,int(h*0.92)))
                    w2,h2=im.size
                    max_side=max(w2,h2)
                    pad=( (max_side-w2)//2, (max_side-h2)//2, (max_side-w2)-(max_side-w2)//2, (max_side-h2)-(max_side-h2)//2 )
                    im=ImageOps.expand(im, pad, fill=CREAM)
                    im=im.resize((256,256), Image.LANCZOS)
                    ax.imshow(im)
                    ax.set_title(f"{p.name[:16]}\n{p.stat().st_size//1024}KB", fontsize=6)
            except Exception as e:
                print(f"err {e}")
                ax.text(0.5,0.5,str(e)[:20],ha='center')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved {out_path}")

grid(top16, out/"grid_top16_plates.png", "Top 16 largest — likely plates (2 vols)")
grid(small16, out/"grid_small16_text.png", "Smallest 16 — likely text/noise (2 vols)")

# histogram quick (no image open)
sizes=[p.stat().st_size//1024 for p in files]
plt.figure(figsize=(6,3))
plt.hist(sizes, bins=30, color='teal', alpha=0.7)
plt.title("JP2 filesize distribution (KB) — 2 vols, 380 images")
plt.xlabel("KB"); plt.ylabel("count")
plt.axvline(500, color='red', linestyle='--', label='500KB thresh ~33% plates')
plt.legend()
plt.tight_layout()
plt.savefig(out/"filesize_hist.png", dpi=150)
plt.close()
print("hist saved")

# also save 8 samples to data/samples
samples=Path("data/samples")
samples.mkdir(parents=True, exist_ok=True)
for i,p in enumerate(top16[:8]):
    with Image.open(p) as im:
        if im.mode!="RGB": im=im.convert("RGB")
        w,h=im.size
        im=im.crop((0,0,w,int(h*0.92)))
        w2,h2=im.size
        max_side=max(w2,h2)
        pad=( (max_side-w2)//2, (max_side-h2)//2, (max_side-w2)-(max_side-w2)//2, (max_side-h2)-(max_side-h2)//2 )
        im=ImageOps.expand(im, pad, fill=CREAM)
        im.thumbnail((512,512), Image.LANCZOS)
        im.save(samples/f"sample_{i:02d}.jpg", quality=92)
        print(f"sample {i} saved")
print("done")
