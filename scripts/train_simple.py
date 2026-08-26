#!/usr/bin/env python3
import yaml, torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import argparse

class DS(Dataset):
    def __init__(self, root, res):
        self.files=list(Path(root).glob("*.jpg")) + list(Path(root).glob("*.png"))
        self.t=transforms.Compose([transforms.Resize(res, antialias=True),transforms.CenterCrop(res),transforms.RandomHorizontalFlip(0.5),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        return self.t(Image.open(self.files[i]).convert("RGB"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args=parser.parse_args()
    with open(args.config) as f:
        cfg=yaml.safe_load(f)
    res=cfg["training"]["resolution"]
    ds=DS(cfg["data"]["train_data_dir"], res)
    dl=DataLoader(ds, batch_size=cfg["training"]["train_batch_size"], shuffle=True, num_workers=0)
    model=UNet2DModel(
        sample_size=cfg["model"]["sample_size"],
        in_channels=3,out_channels=3,
        layers_per_block=cfg["model"]["layers_per_block"],
        block_out_channels=tuple(cfg["model"]["block_out_channels"]),
        down_block_types=tuple(cfg["model"]["down_block_types"]),
        up_block_types=tuple(cfg["model"]["up_block_types"]),
        dropout=cfg["model"].get("dropout",0.1)
    )
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    model.to(device)
    scheduler=DDPMScheduler(num_train_timesteps=cfg["diffusion"]["num_train_timesteps"], beta_schedule=cfg["diffusion"]["beta_schedule"])
    optimizer=torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], betas=(cfg["training"].get("adam_beta1",0.95), cfg["training"].get("adam_beta2",0.999)), weight_decay=cfg["training"].get("adam_weight_decay",1e-6))
    lr_sched=get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg["training"].get("lr_warmup_steps",100), num_training_steps=cfg["training"]["max_train_steps"])
    max_steps=cfg["training"]["max_train_steps"]
    out=Path(cfg["output"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # save config
    import json, yaml as y
    with open(out/"config.yaml","w") as f:
        y.dump(cfg,f)

    model.train()
    global_step=0
    pbar=tqdm(total=max_steps)
    # infinite loader
    import itertools, random
    while global_step < max_steps:
        for batch in dl:
            batch=batch.to(device)
            noise=torch.randn_like(batch)
            timesteps=torch.randint(0,1000,(batch.shape[0],), device=device).long()
            noisy=scheduler.add_noise(batch, noise, timesteps)
            pred=model(noisy, timesteps).sample
            loss=torch.nn.functional.mse_loss(pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()
            global_step+=1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_sched.get_last_lr()[0]:.2e}")
            if global_step%250==0:
                # sample
                model.eval()
                try:
                    pipeline=DDPMPipeline(unet=model, scheduler=scheduler)
                    pipeline.to(device)
                    # pipeline expects cpu? use device
                    images=pipeline(batch_size=4, num_inference_steps=20, output_type="pil").images
                    # save grid
                    w,h=images[0].size
                    grid=Image.new("RGB",(w*2,h*2))
                    for idx,img in enumerate(images[:4]):
                        grid.paste(img, ((idx%2)*w, (idx//2)*h))
                    grid.save(out/f"samples-{global_step:06d}.png")
                    print(f"saved samples {global_step}")
                except Exception as e:
                    print(f"sample failed {e}")
                model.train()
            if global_step%500==0:
                # checkpoint
                ckpt=out/f"checkpoint-{global_step}"
                ckpt.mkdir(exist_ok=True)
                model.save_pretrained(ckpt/"unet")
                torch.save(optimizer.state_dict(), ckpt/"optimizer.pt")
                print(f"saved ckpt {ckpt}")
            if global_step>=max_steps:
                break
        # end epoch
    # final
    model.save_pretrained(out/"final"/"unet")
    print("done")

if __name__=="__main__":
    main()
