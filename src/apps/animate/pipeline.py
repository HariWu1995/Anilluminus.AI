"""
Reference:
    https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff
    https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/animatediff.md
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

if gpu_free_memory < 10:
    # Customized Pipeline for low-VRAM: tested on 6Gb GPU
    from .pipelines import AnimateDiffSparseControlNetPipeline
else:
    from diffusers import AnimateDiffSparseControlNetPipeline

from diffusers.models import AutoencoderKL, MotionAdapter, SparseControlNetModel as ControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler as Scheduler
from diffusers.loaders import FromSingleFileMixin
from diffusers.utils import export_to_video, load_video
from diffusers.utils.import_utils import is_xformers_available


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32


class AnimationPipeline(AnimateDiffSparseControlNetPipeline, FromSingleFileMixin):
    pass


def load_pipeline(
        sd_version: str,
        model_path: str,
          vae_path: str,
        motion_adapter_path: str,
        lora_adapter_path: str,
        controlnet_path: str,
    ):

    # model_path = "SG161222/Realistic_Vision_V5.1_noVAE"
    # vae_path = "stabilityai/sd-vae-ft-mse"
    # controlnet_path = "guoyww/animatediff-sparsectrl-rgb"
    # lora_adapter_path = "guoyww/animatediff-motion-lora-v1-5-3"
    # motion_adapter_path = "guoyww/animatediff-motion-adapter-v1-5-3"

    torch.cuda.empty_cache()

    # Validate GPU memory for SD
    is_enough_memory = validate_gpu_memory(sd_version)
    
    if is_enough_memory:
        device = DEVICE
        dtype = DTYPE
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    # Load checkpoints
    autoencoder = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype).to(device)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype).to(device)
    motion_adapter = MotionAdapter.from_pretrained(motion_adapter_path, torch_dtype=dtype).to(device)

    scheduler_path = './checkpoints'
    scheduler = Scheduler.from_pretrained(scheduler_path, subfolder="scheduler", 
                                                        beta_schedule="linear",
                                                        algorithm_type="dpmsolver++",
                                                        use_karras_sigmas=True)
    config = dict(
        motion_adapter = motion_adapter,
            controlnet = controlnet,
                   vae = autoencoder,
             scheduler = scheduler,
           torch_dtype = dtype,
    )

    if model_path.endswith(tuple(MODEL_EXTENSIONS)):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = AnimationPipeline.from_single_file(model_path, **config).to(device)
    else:
        pipe = AnimationPipeline.from_pretrained(model_path, **config).to(device)
    # pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)

    if lora_adapter_path.endswith(tuple(MODEL_EXTENSIONS)):
        lora_dir, ckpt_name = os.path.split(lora_adapter_path)
        pipe.load_lora_weights(lora_dir, weight_name=ckpt_name, adapter_name="motion_lora")
    else:
        pipe.load_lora_weights(lora_adapter_path, adapter_name="motion_lora")
    
    pipe.enable_lora()
    pipe.fuse_lora(lora_scale=1.0)

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    # pipe.enable_model_cpu_offload()

    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()

    return pipe


def run_pipeline(
        sd_version: str, 
        model_path: str, 
        vae_path: str, 
        lora_path: str, 
        adapter_path: str, 
        ctrlnet_path: str,
        lucky_number: int = None,
        **kwargs
    ):

    if not lucky_number:
        lucky_number = rd.randint(1, 1995)
    kwargs['generator'] = torch.Generator().manual_seed(lucky_number)

    pipe = load_pipeline(sd_version, model_path, vae_path, adapter_path, lora_path, ctrlnet_path)
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

