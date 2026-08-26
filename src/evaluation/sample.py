#!/usr/bin/env python3
"""
Sample from checkpoint, generate grids
Usage: python src/evaluation/sample.py --checkpoint experiments/exp001_baseline_128/checkpoint-50000 --output results/samples
"""
import argparse
from pathlib import Path
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/samples")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    # try to load unet_ema or unet
    unet_path = ckpt / "unet_ema"
    if not unet_path.exists():
        unet_path = ckpt / "unet"
    if not unet_path.exists():
        unet_path = ckpt  # maybe contains model directly
    print(f"loading unet from {unet_path}")
    unet = UNet2DModel.from_pretrained(unet_path, torch_dtype=torch.float16)
    scheduler = DDPMScheduler.from_pretrained(unet_path) if (unet_path / "scheduler").exists() else DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet.to(device)
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    images = pipeline(batch_size=args.num, num_inference_steps=args.steps).images
    # images are PIL if output_type default
    # if numpy, convert
    if isinstance(images[0], Image.Image):
        pil_images = images
    else:
        pil_images = [Image.fromarray((img*255).astype("uint8")) for img in images]

    for i, im in enumerate(pil_images):
        im.save(out / f"sample_{i:04d}.png")
    # grid
    grid_size = int(args.num**0.5)
    w,h = pil_images[0].size
    cols = 4
    rows = (len(pil_images)+cols-1)//cols
    grid = Image.new("RGB", (w*cols, h*rows), (250,246,232))
    for idx, im in enumerate(pil_images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(im, (x,y))
    grid.save(out / "grid.png")
    print(f"saved {len(pil_images)} to {out}")

if __name__ == "__main__":
    main()
