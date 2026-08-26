#!/usr/bin/env python3
"""
Download Curtis's Botanical Magazine plates from Archive.org
Enumerates Curtis volumes, downloads jp2.zip, extracts likely plate images.

Usage:
  python scripts/download_curtis.py --sample 3 --output data/raw/curtis
  python scripts/download_curtis.py --identifiers curtissflowergar0000unse s1id13292280 --output data/raw/curtis
  python scripts/download_curtis.py --all --output data/raw/curtis --max-volumes 100
"""
import argparse, json, os, sys, zipfile, io, re, time
from pathlib import Path
import requests
from tqdm import tqdm

# Known Curtis identifiers from research
# s1id13292120.. etc are 47 vols, plus mobot, plus curtisflower reprint
CURTIS_SEED_IDS = [
    "curtissflowergar0000unse",  # 120 plates curated 1787-1807, easiest start
    "s1id13292120", "s1id13292130", "s1id13292140", "s1id13292150", "s1id13292160",
    "s1id13292200", "s1id13292210", "s1id13292220", "s1id13292230", "s1id13292240",
    "s1id13292250", "s1id13292260", "s1id13292270", "s1id13292280", "s1id13292290",
]

ARCHIVE_SEARCH = 'https://archive.org/advancedsearch.php?q=title:"Curtis\'s+botanical+magazine"+AND+mediatype:texts&fl=identifier,year,title,imagecount&rows=200&page=1&output=json'

def search_archive():
    print(f"[search] querying {ARCHIVE_SEARCH}")
    r = requests.get(ARCHIVE_SEARCH, timeout=30)
    r.raise_for_status()
    docs = r.json()["response"]["docs"]
    # Filter to likely Curtis magazine (exclude some non-magazine hits)
    filtered = [d for d in docs if "curtis" in d["identifier"].lower() or "botanical" in d["title"].lower()]
    print(f"[search] found {len(docs)} total, {len(filtered)} likely Curtis")
    for d in filtered[:10]:
        print(f"  - {d['identifier']} ({d.get('year','?')}) imagecount={d.get('imagecount','?')}")
    return filtered

def download_identifier(identifier, out_dir: Path, extract=True, max_images=None):
    """Download jp2.zip and optionally extract images"""
    out_dir = Path(out_dir) / identifier
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_url = f"https://archive.org/download/{identifier}/{identifier}_jp2.zip"
    zip_path = out_dir / f"{identifier}_jp2.zip"
    if zip_path.exists():
        print(f"[skip] {zip_path} already exists")
    else:
        print(f"[download] {identifier} -> {zip_url}")
        try:
            with requests.get(zip_url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    print(f"  [404] no jp2.zip for {identifier}, trying alternate...")
                    # try *_jp2.zip with different naming or fallback to images
                    return None
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=identifier) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"  [error] download failed for {identifier}: {e}")
            if zip_path.exists():
                zip_path.unlink()
            return None
    if extract and zip_path.exists():
        extract_dir = out_dir / "jp2"
        extract_dir.mkdir(exist_ok=True)
        # Count already extracted
        existing = list(extract_dir.glob("*.jp2"))
        if existing:
            print(f"  [extract] already {len(existing)} jp2 extracted")
            return extract_dir
        print(f"  [extract] unzipping {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                members = z.namelist()
                if max_images:
                    members = members[:max_images]
                for m in tqdm(members, desc="extract"):
                    z.extract(m, extract_dir)
            print(f"  [extract] done -> {extract_dir} ({len(list(extract_dir.rglob('*.jp2')))} files)")
        except Exception as e:
            print(f"  [extract error] {e}")
            return None
        return extract_dir
    return out_dir

def main():
    parser = argparse.ArgumentParser(description="Download Curtis plates")
    parser.add_argument("--output", default="data/raw/curtis", help="output dir")
    parser.add_argument("--identifiers", nargs="*", help="specific archive.org identifiers")
    parser.add_argument("--sample", type=int, help="download N random Curtis volumes from search")
    parser.add_argument("--all", action="store_true", help="download all Curtis volumes from search")
    parser.add_argument("--max-volumes", type=int, default=100, help="max volumes when using --all")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.identifiers:
        ids = args.identifiers
    elif args.sample:
        docs = search_archive()
        import random
        random.seed(42)
        # prioritize s1id/mobot/curtis
        prioritized = [d["identifier"] for d in docs if any(p in d["identifier"] for p in ["s1id", "mobot", "curtis"])]
        if not prioritized:
            prioritized = [d["identifier"] for d in docs]
        ids = random.sample(prioritized, min(args.sample, len(prioritized)))
        print(f"[sample] chosen: {ids}")
    elif args.all:
        docs = search_archive()
        prioritized = [d["identifier"] for d in docs if any(p in d["identifier"] for p in ["s1id", "mobot", "curtis"])]
        ids = prioritized[:args.max_volumes]
        print(f"[all] downloading {len(ids)} volumes")
    else:
        # default: seed + 2 more for quick start
        ids = CURTIS_SEED_IDS[:3]
        print(f"[default] using seed ids: {ids}")

    # Also add fallback for quick 120-plate reprint if not included
    if "curtissflowergar0000unse" not in ids and len(ids) < 5:
        ids = ["curtissflowergar0000unse"] + ids

    manifest = []
    for ident in ids:
        # be nice to archive.org
        time.sleep(1)
        result = download_identifier(ident, out, extract=not args.no_extract)
        manifest.append({"identifier": ident, "path": str(result) if result else None, "status": "ok" if result else "failed"})

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] manifest -> {manifest_path}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
