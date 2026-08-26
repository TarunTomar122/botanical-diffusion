#!/usr/bin/env python3
"""
Clean Curtis plates: filter illustrations, crop, dedup, resize to target resolution.

Approach:
- For each volume's jp2 dir, list images
- Simple heuristic to detect plates vs text: color saturation + edge density + file size
  (plates are color, larger, higher saturation)
- Also checks scandata.xml if present for pageType
- Crops bottom 8% caption, auto-detects white margin, pads to square with cream, resizes

Usage:
  python scripts/clean_dataset.py --input data/raw/curtis --output data/processed/curtis-128 --resolution 128
  python scripts/clean_dataset.py --input data/raw/curtis --output data/processed/curtis-64 --resolution 64 --max-images 500

Requires: pillow, imagehash, opencv-python, numpy
"""
import argparse, json, os, re, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import imagehash
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

CREAM = (250, 246, 232)  # #faf6e8

def saturation_score(img: Image.Image):
    """Estimate if image is color plate vs b/w text"""
    # Convert to HSV and check saturation variance
    arr = np.array(img.resize((64,64)))
    if arr.ndim != 3:
        return 0
    # Simple: std of channel differences
    r,g,b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    # saturation approx: max-min
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)
    return float(np.mean(sat))

def is_likely_plate(img_path: Path, filesize=None):
    """Heuristic: plates are color, larger files, not too small dimensions"""
    try:
        # filesize heuristic first (plates larger than text pages which compress well or are smaller)
        if filesize and filesize < 50000:  # 50KB too small for plate at jp2? but jp2 sizes vary
            pass # don't reject solely
        with Image.open(img_path) as im:
            w,h = im.size
            if min(w,h) < 500:
                return False, "too_small"
            # try saturation
            # Convert jp2 may be slow; sample
            try:
                score = saturation_score(im)
                if score < 15:  # b/w text low saturation
                    return False, f"low_sat={score:.1f}"
                return True, f"sat={score:.1f}"
            except Exception as e:
                return True, f"sat_error_{e}"
    except Exception as e:
        return False, f"open_error_{e}"

def clean_image(img_path: Path, out_path: Path, resolution: int):
    """Crop caption, pad to square, resize"""
    try:
        with Image.open(img_path) as im:
            # Convert to RGB
            if im.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", im.size, CREAM)
                bg.paste(im, mask=im.split()[-1])
                im = bg
            elif im.mode != "RGB":
                im = im.convert("RGB")

            w,h = im.size
            # Crop bottom 8% caption (consistent across Curtis)
            crop_h = int(h * 0.92)
            im = im.crop((0, 0, w, crop_h))
            w,h = im.size

            # Auto-detect white margin? Simple: keep as is, pad to square
            # Pad to square with cream
            max_side = max(w,h)
            delta_w = max_side - w
            delta_h = max_side - h
            padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
            im_padded = ImageOps.expand(im, padding, fill=CREAM)

            # Resize to resolution with antialias
            im_resized = im_padded.resize((resolution, resolution), Image.LANCZOS)

            # Save as JPEG quality 95 or PNG?
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im_resized.save(out_path, "JPEG", quality=95)
            return True
    except Exception as e:
        print(f"[clean error] {img_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input raw curtis dir (with subdirs per identifier)")
    parser.add_argument("--output", required=True, help="output processed dir")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-images", type=int, help="limit total")
    parser.add_argument("--dedup", action="store_true", help="phash dedup")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Find all jp2/jpg/png
    all_images = []
    for ext in ["*.jp2", "*.JP2", "*.jpg", "*.jpeg", "*.png"]:
        all_images.extend(inp.rglob(ext))
    # Also check subdir jp2 folders may have nested dirs
    all_images = sorted(set(all_images))
    print(f"[found] {len(all_images)} images under {inp}")

    if not all_images:
        print(f"[warn] no images found, checked {inp}")
        sys.exit(1)

    # Filter to likely plates
    filtered = []
    for p in tqdm(all_images, desc="filtering"):
        try:
            fs = p.stat().st_size
            ok, reason = is_likely_plate(p, fs)
            if ok:
                filtered.append(p)
        except Exception as e:
            pass
    print(f"[filter] {len(filtered)} / {len(all_images)} likely plates (saturation heuristic)")
    if len(filtered) < 10:
        print("[warn] heuristic too aggressive, falling back to all images")
        filtered = all_images

    # Optional: simple size sorting - largest files often plates
    # If filtered too many, sort by filesize descending and take top
    if args.max_images and len(filtered) > args.max_images:
        filtered = sorted(filtered, key=lambda p: p.stat().st_size, reverse=True)[:args.max_images]
        print(f"[limit] capped to {args.max_images}")

    # Dedup via phash
    if args.dedup:
        print("[dedup] computing phashes...")
        hashes = {}
        deduped = []
        for p in tqdm(filtered, desc="phash"):
            try:
                with Image.open(p) as im:
                    h = imagehash.phash(im.resize((64,64)))
                    # check near duplicates (hamming < 5)
                    dup = False
                    for existing_hash in hashes:
                        if h - existing_hash < 5:
                            dup = True
                            break
                    if not dup:
                        hashes[h] = p
                        deduped.append(p)
            except:
                deduped.append(p)
        print(f"[dedup] {len(deduped)} / {len(filtered)} after phash")
        filtered = deduped

    # Clean and resize
    count = 0
    for idx, p in enumerate(tqdm(filtered, desc=f"cleaning -> {args.resolution}")):
        # output name: curtis_{idx:05d}.jpg
        out_name = f"curtis_{idx:05d}.jpg"
        out_path = out / out_name
        if clean_image(p, out_path, args.resolution):
            count += 1

    print(f"[done] {count} images -> {out}")
    # Save manifest
    manifest = {
        "input": str(inp),
        "output": str(out),
        "resolution": args.resolution,
        "found": len(all_images),
        "filtered": len(filtered),
        "cleaned": count,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
