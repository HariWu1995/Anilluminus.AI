from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np
import torch

from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.import_utils import is_xformers_available

from src.utils import MODEL_EXTENSIONS
from src.utils import profile_single_gpu, validate_gpu_memory


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32


def load_pipeline(model_path: str, sd_version: str, num_in_channels: int = 9):
        
    torch.cuda.empty_cache()

    # Validate GPU memory for SD-XL
    is_enough_memory = validate_gpu_memory(sd_version)
    
    if is_enough_memory:
        device = DEVICE
        dtype = DTYPE
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    if sd_version == 'SD-15':
        from diffusers import StableDiffusionInpaintPipeline as SdInpaintPipeline

    elif sd_version == 'SD-XL':
        from diffusers import StableDiffusionXLInpaintPipeline as SdInpaintPipeline

    config = dict(torch_dtype=dtype, num_in_channels=num_in_channels, local_files_only=True)

    if model_path.endswith(tuple(MODEL_EXTENSIONS)):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = SdInpaintPipeline.from_single_file(model_path, **config).to(device)
    else:
        pipe = SdInpaintPipeline.from_pretrained(model_path, **config).to(device)

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()
    # pipe.enable_model_cpu_offload()

    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()

    return pipe


def run_pipeline(
    model_path: str, 
    sd_version: str, 
    n_channels: int,
    image: ImageClass, 
    mask: ImageClass, 
    prompt: str = '', 
    nrompt: str = '', 
    batch_size: int = 1,
    **kwargs
):

    pipe = load_pipeline(model_path, sd_version, n_channels)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    
    W, H = image.size

    diffusion_kwargs = dict(height = H, width = W)
    diffusion_kwargs.update(kwargs)

    all_generated = []

    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")
    
        generated = pipe(prompt = prompt, 
                negative_prompt = nrompt, 
                         image = image, 
                    mask_image = mask, **diffusion_kwargs).images[0]
        
        all_generated.append(generated)

    return all_generated


