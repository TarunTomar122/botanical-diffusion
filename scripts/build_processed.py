#!/usr/bin/env python3
"""
Build processed dataset from raw Curtis jp2: pick largest files per volume, crop, pad, resize.

Fast heuristic: per volume, sort by filesize descending, take top_k or >thresh.

Usage:
  python scripts/build_processed.py --input data/raw/curtis --output data/processed/curtis-128 --resolution 128 --top-per-volume 40
  python scripts/build_processed.py --input data/raw/curtis --output data/processed/curtis-64 --resolution 64 --thresh-kb 500
"""
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import json

CREAM=(250,246,232)

def process_one(in_path: Path, out_path: Path, resolution: int):
    try:
        with Image.open(in_path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            w,h = im.size
            # crop bottom 8% (caption)
            im = im.crop((0,0,w,int(h*0.92)))
            w,h = im.size
            # pad to square
            max_side = max(w,h)
            pad_w = max_side - w
            pad_h = max_side - h
            padding = (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2)
            im = ImageOps.expand(im, padding, fill=CREAM)
            im = im.resize((resolution,resolution), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, "JPEG", quality=95)
            return True
    except Exception as e:
        print(f"err {in_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--top-per-volume", type=int, default=40, help="take N largest per volume dir")
    parser.add_argument("--thresh-kb", type=int, help="alternative: filesize threshold KB")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Find volume dirs: each subdir under inp with jp2 subfolder
    volumes = sorted([d for d in inp.iterdir() if d.is_dir()])
    print(f"volumes: {[v.name for v in volumes]}")

    selected = []
    for vol in volumes:
        # find all jp2 under vol
        jp2s = list(vol.rglob("*.jp2"))
        if not jp2s:
            continue
        jp2s_sorted = sorted(jp2s, key=lambda p: p.stat().st_size, reverse=True)
        if args.thresh_kb:
            picks = [p for p in jp2s_sorted if p.stat().st_size > args.thresh_kb*1024]
            print(f"{vol.name}: {len(jp2s)} total, {len(picks)} >{args.thresh_kb}KB")
        else:
            picks = jp2s_sorted[:args.top_per_volume]
            print(f"{vol.name}: {len(jp2s)} total, taking top {len(picks)}")
        selected.extend(picks)

    print(f"selected total {len(selected)} images")
    # process
    count=0
    for idx, p in enumerate(tqdm(selected, desc=f"processing {args.resolution}")):
        out_path = out / f"curtis_{idx:05d}.jpg"
        if process_one(p, out_path, args.resolution):
            count+=1
    print(f"done: {count} -> {out}")
    manifest = {"input": str(inp), "output": str(out), "resolution": args.resolution, "selected": len(selected), "processed": count, "top_per_volume": args.top_per_volume, "thresh_kb": args.thresh_kb}
    with open(out / "manifest.json","w") as f:
        json.dump(manifest,f,indent=2)
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
