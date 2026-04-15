from diffusers import AutoencoderDC, SanaTransformer2DModel
import torch
from safetensors.torch import load_file
import json
import os
from blip3o.utils import rank0_print


def build_sana(vision_tower_cfg, **kwargs):
    rank0_print(f"build_sana: vision_tower_cfg.diffusion_name_or_path: {vision_tower_cfg.diffusion_name_or_path}")
    sana = SanaTransformer2DModel.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    return sana


def build_vae(vision_tower_cfg, **kwargs):
    rank0_print(f"build_vae: vision_tower_cfg.diffusion_name_or_path: {vision_tower_cfg.diffusion_name_or_path}")
    vae = AutoencoderDC.from_pretrained(vision_tower_cfg.diffusion_name_or_path, subfolder="vae", torch_dtype=torch.bfloat16)
    return vae


