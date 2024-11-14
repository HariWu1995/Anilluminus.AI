"""
Reference:
    https://github.com/alibaba/animate-anything
"""
import os, gc

from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import random as rd
import numpy as np
import torch

from src.utils import MODEL_EXTENSIONS, GigaValue
from src.utils import profile_single_gpu, validate_gpu_memory

gpu_profile = profile_single_gpu(device_id=0)
gpu_free_memory = round(gpu_profile[-1] / GigaValue, 2)

# Customized Pipeline for low-VRAM: tested on 6Gb GPU
from .pipelines import LatentToVideoPipeline as AnimationPipeline

from diffusers.utils import export_to_video, load_video
from diffusers.utils.import_utils import is_xformers_available


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32


def load_pipeline(model_path: str):

    torch.cuda.empty_cache()

    # Validate GPU memory for SD
    is_enough_memory = validate_gpu_memory(sd_version)
    
    if is_enough_memory:
        device = DEVICE
        dtype = DTYPE
        config = dict(torch_dtype=dtype, variant="fp16")
    else:
        device = torch.device('cpu')
        dtype = torch.float32
        config = dict(torch_dtype=dtype)

    # Load checkpoints
    pipe = AnimationPipeline.from_pretrained(model_path, **kwargs).to(device)

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    # pipe.enable_model_cpu_offload()

    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()

    return pipe


def run_pipeline(model_path: str, lucky_number: int = None, **kwargs):

    if not lucky_number:
        lucky_number = rd.randint(1, 1995)
    kwargs['generator'] = torch.Generator().manual_seed(lucky_number)

    pipe = load_pipeline(model_path)
    video = pipe(**kwargs).frames[0]

    return video


if __name__ == "__main__":

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    vae_path = "D:/stable-diffusion/sd-15/VAE/stabilityai-sd-vae-ft-mse"
    model_path = "D:/stable-diffusion/sd-15/checkpoints/absolutereality_v16.safetensors"
    # model_path = "D:/stable-diffusion/sd-15/checkpoints/realisticVisionV51_v20_woVAE.safetensors"

    animatediff_dir = "E:/stable-diffusion/AnimateDiff"
    controlnet_path = f"{animatediff_dir}/controlnet/animatediff-sparsectrl-rgb"
    lora_adapter_path = f"{animatediff_dir}/lora/animatediff-motion-sd15-v3"
    motion_adapter_path = f"{animatediff_dir}/adapter/animatediff-motion-sd15-v3"

    image_path = './tests/vfst/styles/Screenshot 2024-10-01 114518.png'
    image = Image.open(image_path).convert('RGB').resize((620, 350))

    video = run_pipeline(

        sd_version = 'SD-15',
        model_path = model_path,
        vae_path = vae_path,
        lora_path = lora_adapter_path,
        adapter_path = motion_adapter_path,
        ctrlnet_path = controlnet_path,

        prompt = 'car running',
        conditioning_frames = image,
        controlnet_frame_indices = [0],

        num_videos_per_prompt = 1,
        num_inference_steps = 12,
        num_frames = 15,
    )

    _ = export_to_video(video, fps=10, output_video_path="logs/video.mp4")

