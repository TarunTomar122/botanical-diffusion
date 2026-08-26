#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import json

CREAM=(250,246,232)

def is_plate(p, sat_thresh=15):
    try:
        im = Image.open(p).convert("RGB").resize((64,64))
        a = np.array(im)
        sat = (a.max(axis=2)-a.min(axis=2)).mean()
        return sat > sat_thresh, sat
    except:
        return False, 0

def process_one(in_path, out_path, res):
    try:
        with Image.open(in_path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            w,h = im.size
            im = im.crop((0,0,w,int(h*0.92)))
            w,h = im.size
            max_side = max(w,h)
            pad = ((max_side-w)//2, (max_side-h)//2, (max_side-w)-(max_side-w)//2, (max_side-h)-(max_side-h)//2)
            im = ImageOps.expand(im, pad, fill=CREAM)
            im = im.resize((res,res), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, "JPEG", quality=95)
            return True
    except Exception as e:
        print(f"err {in_path}: {e}")
        return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--top-per-volume", type=int, default=40)
    parser.add_argument("--sat-thresh", type=float, default=15)
    args=parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    volumes = sorted([d for d in inp.iterdir() if d.is_dir()])
    selected=[]
    for vol in volumes:
        jp2s = list(vol.rglob("*.jp2"))
        if not jp2s: continue
        jp2s_sorted = sorted(jp2s, key=lambda p: p.stat().st_size, reverse=True)
        # Take top 60 per vol, then filter by sat, keep 40
        candidates = jp2s_sorted[:60]
        plates=[]
        for p in candidates:
            ok, sat = is_plate(p, args.sat_thresh)
            if ok:
                plates.append(p)
            if len(plates) >= args.top_per_volume:
                break
        print(f"{vol.name}: {len(jp2s)} total, top60 filtered to {len(plates)} plates (sat>{args.sat_thresh})")
        selected.extend(plates)

    print(f"selected total {len(selected)}")
    count=0
    for idx, p in enumerate(tqdm(selected, desc=f"processing {args.resolution}")):
        if process_one(p, out / f"curtis_{idx:05d}.jpg", args.resolution):
            count+=1
    print(f"done {count} -> {out}")
    with open(out / "manifest.json","w") as f:
        json.dump({"selected": len(selected), "processed": count, "sat_thresh": args.sat_thresh}, f, indent=2)

if __name__=="__main__":
    main()
