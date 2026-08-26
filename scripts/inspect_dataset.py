#!/usr/bin/env python3
"""
Inspect dataset: generate grids, stats, samples.

Usage:
  python scripts/inspect_dataset.py --data data/processed/curtis-128 --output figures/dataset-inspection
  python scripts/inspect_dataset.py --data data/raw/curtis --output figures/raw-inspection --raw
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def grid(images, out_path, n=16, title=""):
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < len(images):
            try:
                im = Image.open(images[i]).convert("RGB")
                ax.imshow(im)
                ax.set_title(Path(images[i]).name[:20], fontsize=6)
            except:
                ax.text(0.5,0.5,"ERR",ha='center')
        else:
            ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[grid] {out_path}")

def stats(images, out_dir: Path):
    dims = []
    aspects = []
    sizes = []
    for p in images:
        try:
            with Image.open(p) as im:
                w,h = im.size
                dims.append((w,h))
                aspects.append(w/h)
                sizes.append(Path(p).stat().st_size/1024)
        except:
            pass
    if not dims:
        return
    ws, hs = zip(*dims)
    print(f"[stats] {len(dims)} images")
    print(f"  dims: {min(ws)}x{min(hs)} to {max(ws)}x{max(hs)}")
    print(f"  mean: {np.mean(ws):.0f}x{np.mean(hs):.0f}")
    print(f"  aspect: {np.mean(aspects):.2f} ± {np.std(aspects):.2f} (1.0=square, >1 wide)")
    print(f"  filesize KB: {np.mean(sizes):.1f} ± {np.std(sizes):.1f}")

    # plots
    fig, axes = plt.subplots(1,3, figsize=(12,3))
    axes[0].hist(ws, bins=20, color='teal', alpha=0.7)
    axes[0].set_title("Width")
    axes[1].hist(hs, bins=20, color='coral', alpha=0.7)
    axes[1].set_title("Height")
    axes[2].hist(aspects, bins=20, color='slateblue', alpha=0.7)
    axes[2].set_title("Aspect ratio (w/h)")
    plt.tight_layout()
    plt.savefig(out_dir / "dimension_stats.png", dpi=150)
    plt.close()
    # save json
    data = {
        "count": len(dims),
        "width": {"min": int(min(ws)), "max": int(max(ws)), "mean": float(np.mean(ws))},
        "height": {"min": int(min(hs)), "max": int(max(hs)), "mean": float(np.mean(hs))},
        "aspect": {"mean": float(np.mean(aspects)), "std": float(np.std(aspects))},
        "filesize_kb": {"mean": float(np.mean(sizes)), "std": float(np.std(sizes))},
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw", action="store_true", help="data is raw jp2 nested")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # find images
    exts = ["*.jpg","*.jpeg","*.png","*.JPG","*.jp2","*.JP2"]
    images = []
    for e in exts:
        images.extend(data.rglob(e))
    images = sorted(images)
    print(f"[found] {len(images)} images under {data}")
    if not images:
        print("[warn] no images")
        return

    # sample
    random.seed(42)
    n = min(64, len(images))
    sample = random.sample(images, n)

    # grids
    grid(sample[:16], out / "grid_random_16.png", n=16, title="Random 16")
    if len(images) >= 64:
        grid(sample[:64], out / "grid_random_64.png", n=64, title="Random 64")
    # best-looking heuristic: largest files
    largest = sorted(images, key=lambda p: p.stat().st_size, reverse=True)[:16]
    grid(largest, out / "grid_largest_16.png", n=16, title="Largest files (likely plates)")
    smallest = sorted(images, key=lambda p: p.stat().st_size)[:16]
    grid(smallest, out / "grid_smallest_16.png", n=16, title="Smallest files (likely text/noise)")

    # copy 8 beautiful samples to data/samples for README
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(largest[:8]):
        try:
            im = Image.open(p).convert("RGB")
            # resize to 512 for samples
            im.thumbnail((512,512))
            im.save(samples_dir / f"sample_{i:02d}.jpg", quality=90)
        except: pass
    print(f"[samples] saved 8 to {samples_dir}")

    # stats
    stats(images, out)

    # also save a markdown snippet
    with open(out / "README.md","w") as f:
        f.write(f"# Dataset Inspection: {data}\n\n")
        f.write(f"- **Count:** {len(images)}\n")
        f.write(f"- **Path:** {data}\n\n")
        f.write("![random 16](grid_random_16.png)\n\n")
        f.write("![largest 16](grid_largest_16.png)\n\n")
        f.write("![smallest 16](grid_smallest_16.png)\n\n")
        f.write("![stats](dimension_stats.png)\n")
    print(f"[done] -> {out}")

if __name__ == "__main__":
    main()
