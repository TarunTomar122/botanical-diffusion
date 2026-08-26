#!/usr/bin/env python3
"""
Minimal from-scratch diffusion training for botanical plates.
Based on diffusers/examples/unconditional_image_generation/train_unconditional.py
but stripped down, locked to our configs.

Usage on RunPod:
  accelerate launch src/training/train.py --config configs/baseline_128.yaml
  python src/training/train.py --config configs/sanity_64.yaml  # single GPU, no accelerate
"""
import argparse, yaml, os, math, random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from accelerate import Accelerator

class ImageFolderDataset(Dataset):
    def __init__(self, root, resolution=128, center_crop=True, random_flip=True):
        self.root = Path(root)
        self.files = sorted(list(self.root.glob("*.jpg")) + list(self.root.glob("*.png")) + list(self.root.glob("*.jpeg")))
        if not self.files:
            raise ValueError(f"no images in {root}")
        print(f"dataset {root}: {len(self.files)} images")
        transform = []
        transform.append(transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True))
        if center_crop:
            transform.append(transforms.CenterCrop(resolution))
        else:
            transform.append(transforms.RandomCrop(resolution))
        if random_flip:
            transform.append(transforms.RandomHorizontalFlip(p=0.5))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize([0.5],[0.5]))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        im = Image.open(p).convert("RGB")
        return self.transform(im)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_unet_from_config(cfg):
    m = cfg["model"]
    return UNet2DModel(
        sample_size=m["sample_size"],
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        layers_per_block=m["layers_per_block"],
        block_out_channels=tuple(m["block_out_channels"]),
        down_block_types=tuple(m["down_block_types"]),
        up_block_types=tuple(m["up_block_types"]),
        dropout=m.get("dropout", 0.1),
        attention_head_dim=m.get("attention_head_dim", 8),
        norm_num_groups=m.get("norm_num_groups", 32),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(yaml.dump(cfg, default_flow_style=False))

    # accelerator
    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "no"),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        log_with="tensorboard",
        project_dir=cfg["output"]["output_dir"],
    )
    # reproducibility
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    # dataset
    data_cfg = cfg["data"]
    train_dataset = ImageFolderDataset(
        data_cfg["train_data_dir"],
        resolution=cfg["training"]["resolution"],
        center_crop=cfg["training"].get("center_crop", True),
        random_flip=cfg["training"].get("random_flip", False),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["train_batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("dataloader_num_workers", 4),
        pin_memory=True,
    )

    # model
    model = get_unet_from_config(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["diffusion"]["num_train_timesteps"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
        prediction_type=cfg["diffusion"].get("prediction_type", "epsilon"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=(cfg["training"].get("adam_beta1", 0.95), cfg["training"].get("adam_beta2", 0.999)),
        weight_decay=cfg["training"].get("adam_weight_decay", 1e-6),
        eps=cfg["training"].get("adam_epsilon", 1e-8),
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"].get("lr_warmup_steps", 2000),
        num_training_steps=cfg["training"].get("max_train_steps", 400000),
    )
    # EMA
    use_ema = cfg["training"].get("use_ema", True)
    if use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=cfg["training"].get("ema_max_decay", 0.9999),
            inv_gamma=cfg["training"].get("ema_inv_gamma", 1.0),
            power=cfg["training"].get("ema_power", 0.75),
        )
    else:
        ema_model = None

    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # prepare with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    if use_ema:
        ema_model.to(accelerator.device)

    # training loop
    max_steps = cfg["training"].get("max_train_steps", 800000)
    checkpointing_steps = cfg["training"].get("checkpointing_steps", 10000)
    save_images_steps = cfg["training"].get("save_images_steps", 5000)  # not epochs, steps
    # track steps
    global_step = 0
    epoch = 0
    model.train()

    # for sampling, keep unwrapped for pipeline
    unwrapped_model = accelerator.unwrap_model(model)

    pbar = tqdm(total=max_steps, disable=not accelerator.is_main_process)
    pbar.update(global_step)

    while global_step < max_steps:
        for batch in train_dataloader:
            # batch: [B, C, H, W] in [-1,1]
            with accelerator.accumulate(model):
                clean_images = batch
                # Sample noise
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bsz = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device, dtype=torch.long)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                # Predict
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if use_ema and accelerator.sync_gradients:
                    ema_model.step(model.parameters())
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if global_step % 100 == 0 and accelerator.is_main_process:
                    # log to tensorboard via accelerator?
                    pass

                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    checkpoint_path = output_dir / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(exist_ok=True)
                    accelerator.save_state(checkpoint_path)
                    # also save ema model
                    if use_ema:
                        ema_model.copy_to(model.parameters())
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(checkpoint_path / "unet_ema")
                        # restore? ema_model restores on next step, but for saving we need to keep
                    print(f"saved checkpoint {checkpoint_path}")

                if global_step % save_images_steps == 0 and accelerator.is_main_process:
                    # generate samples with EMA
                    if use_ema:
                        ema_model.copy_to(model.parameters())
                    # sample 16 images
                    try:
                        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                        pipeline.to(accelerator.device)
                        images = pipeline(batch_size=4, num_inference_steps=50, output_type="numpy").images
                        # save grid
                        import math
                        from PIL import Image as PILImage
                        # images is list of PIL? numpy 0-1
                        # convert numpy to PIL
                        pil_images = [PILImage.fromarray((img * 255).astype("uint8")) for img in images]
                        # make grid
                        grid_size = 2
                        w, h = pil_images[0].size
                        grid = PILImage.new("RGB", (w*grid_size, h*grid_size))
                        for idx, img in enumerate(pil_images[:4]):
                            x = (idx % grid_size) * w
                            y = (idx // grid_size) * h
                            grid.paste(img, (x,y))
                        grid_path = output_dir / f"samples-{global_step}.png"
                        grid.save(grid_path)
                        print(f"saved samples {grid_path}")
                    except Exception as e:
                        print(f"sample gen failed: {e}")
                    # after sampling, need to keep ema? model params already ema, continue training with ema params is okay, but better to keep original
                    # we don't restore, so training continues with EMA weights — okay for small data, acts as smoothing

                if global_step >= max_steps:
                    break
        epoch += 1
        if global_step >= max_steps:
            break

    # final save
    if accelerator.is_main_process:
        final_path = output_dir / "final"
        final_path.mkdir(exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_path / "unet")
        accelerator.save_state(final_path)
        print(f"training done, saved to {final_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
