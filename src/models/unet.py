"""
UNet for botanical diffusion, thin wrapper around diffusers UNet2DModel.
Configs match research/model-research.md
"""
from diffusers import UNet2DModel

def get_unet_35m_128():
    # 35.7M at 128, 4 blocks, attn at 16
    return UNet2DModel(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 256),
        down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
        dropout=0.1,
    )

def get_unet_small_64():
    return UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(64,128,128,256),
        down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
        dropout=0.1,
    )

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = get_unet_35m_128()
    print(f"35M 128: {count_params(m)/1e6:.1f}M params")
    m2 = get_unet_small_64()
    print(f"small 64: {count_params(m2)/1e6:.1f}M params")
