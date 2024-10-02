from typing import List, Union
from tqdm import tqdm

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np
import torch

from diffusers import UniPCMultistepScheduler as Scheduler
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils.import_utils import is_xformers_available

from src.utils import MODEL_EXTENSIONS
from src.utils.profiling import profile_single_gpu, validate_gpu_memory


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if str(DEVICE).__contains__("cuda") else torch.float32


def load_pipeline(model_path: str, adapter_path: str, iencoder_path: str,
                  sd_version: str, num_in_channels: int = 9):
        
    torch.cuda.empty_cache()

    # Validate GPU memory for SD-XL
    is_enough_memory = validate_gpu_memory(sd_version)
    
    if is_enough_memory:
        device = DEVICE
        dtype = DTYPE
    else:
        device = torch.device('cpu')
        dtype = torch.float32

    # print(is_enough_memory, device, dtype)

    if sd_version == 'SD-15':
        from diffusers import StableDiffusionInpaintPipeline as SdInpaintPipeline
        from src.ip_adapter import IPAdapter
        target_blocks = ["block"]

    elif sd_version == 'SD-XL':
        from diffusers import StableDiffusionXLInpaintPipeline as SdInpaintPipeline
        from src.ip_adapter import IPAdapterXL as IPAdapter
        target_blocks = ["up_blocks.0.attentions.1"]

    config = dict(torch_dtype=dtype, num_in_channels=num_in_channels, local_files_only=True)

    if model_path.endswith(tuple(MODEL_EXTENSIONS)):
        config.update(dict(use_safetensors=True if model_path.endswith(".safetensors") else False))
        pipe = SdInpaintPipeline.from_single_file(model_path, **config).to(device)
    else:
        pipe = SdInpaintPipeline.from_pretrained(model_path, **config).to(device)

    # Re-load scheduler
    pipe.scheduler = Scheduler.from_config(pipe.scheduler.config)

    pipe.enable_vae_tiling()
    # pipe.enable_model_cpu_offload()
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()

    # Load Ip-Adapter
    ip_model = IPAdapter(pipe, iencoder_path, adapter_path, device, target_blocks=target_blocks)

    return ip_model


def run_pipeline(
    model_path: str, 
    adapter_path: str, 
    iencoder_path: str, 
    sd_version: str, 
    n_channels: int,
    image: ImageClass, 
    mask: ImageClass, 
    style: ImageClass,
    batch_size: int = 1,
    adapter_scale: float = 1.99,
    **kwargs
):

    pipe = load_pipeline(model_path, adapter_path, iencoder_path, sd_version, n_channels)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    if isinstance(style, np.ndarray):
        style = Image.fromarray(style)
    style = style.resize(image.size)

    image.save('./logs/image.png')
    mask.save('./logs/mask.png')
    style.save('./logs/style.png')
    
    W, H = image.size

    ## Negative content
    neg_content = None # "a girl"
    neg_content_scale = 0.8

    if neg_content is not None:
        from transformers import CLIPTextModelWithProjection as CLIPTextModel, CLIPTokenizer

        pretrained_clip_model = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        textcoder = CLIPTextModel.from_pretrained(pretrained_clip_model).to(pipe.device, dtype=pipe.dtype)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_clip_model)

        tokens = tokenizer([neg_content], return_tensors='pt').to(pipe.device)
        neg_content_emb = textcoder(**tokens).text_embeds
        neg_content_emb *= neg_content_scale
    else:
        neg_content_emb = None

    ## Generation
    diffusion_kwargs = dict(height = H, width = W)
    diffusion_kwargs.update(kwargs)

    all_generated = []

    progress_bar = tqdm(list(range(batch_size)))
    for i in progress_bar:
        progress_bar.set_description(f"Generating {i+1} / {batch_size} ")

        generated = pipe.generate(
            pil_image = style,
                image = image,
           mask_image = mask,
                scale = adapter_scale, **diffusion_kwargs
        )[0]
        
        all_generated.append(generated)

    return all_generated


