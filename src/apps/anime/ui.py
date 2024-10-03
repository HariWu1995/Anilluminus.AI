import os 
from copy import deepcopy
from PIL import Image

import gradio as gr
import numpy as np

from src.utils import scan_checkpoint_dir, preprocess_image, MODEL_EXTENSIONS
from src.utils import POSITIVE_PROMPT, NEGATIVE_PROMPT


## Variables

SD_VERSION = ['SD-15','SD-XL']
default_sd = 'SD-15'

ckpt_dirs = ['./checkpoints/models']
ckpt_lookup = dict()
for ckpt_dir in ckpt_dirs:
    ckpt_dict = scan_checkpoint_dir(ckpt_dir)
    ckpt_lookup.update(ckpt_dict)

default_model = 'Realistic_Vision_V5.1_noVAE'


vae_dirs = ['./checkpoints/vae']
vae_lookup = dict()
for vae_dir in vae_dirs:
    vae_dict = scan_checkpoint_dir(vae_dir, sub_model=True)
    vae_lookup.update(vae_dict)

default_vae = 'stabilityai-sd-vae-ft-mse'


adapter_dirs = ['./checkpoints/adapters']
adapter_lookup = dict()
for adapter_dir in adapter_dirs:
    adapter_dict = scan_checkpoint_dir(adapter_dir, sub_model=True)
    adapter_lookup.update(adapter_dict)

default_adapter = 'animatediff-motion-adapter-sd15-v3'


lora_dirs = ['./checkpoints/loras']
lora_lookup = dict()
for lora_dir in lora_dirs:
    lora_dict = scan_checkpoint_dir(lora_dir, sub_model=True)
    lora_lookup.update(lora_dict)

default_lora = 'animatediff-motion-lora-sd15-v3'


ctrlnet_dirs = ['./checkpoints/controlnets']
ctrlnet_lookup = dict()
for ctrlnet_dir in ctrlnet_dirs:
    ctrlnet_dict = scan_checkpoint_dir(ctrlnet_dir, sub_model=True)
    ctrlnet_lookup.update(ctrlnet_dict)

default_ctrlnet = 'animatediff-sparsectrl-rgb'


## Pipeline

def generate_video(
        prompt, nrompt, controlled_image,
        ckpt, sd_version, vae, lora, adapter, ctrlnet,
        guidance_scale, num_steps, 
        controlnet_scale, num_frames,
    ):

    ckpt_path = ckpt_lookup[ckpt]
    vae_path = vae_lookup[vae]
    lora_path = lora_lookup[lora]

    adapter_path = adapter_lookup[adapter]
    ctrlnet_path = ctrlnet_lookup[ctrlnet]

    ## Preprocessing
    controlled_image = Image.fromarray(controlled_image).convert('RGB')
    controlled_image, \
            (W, H) = preprocess_image(controlled_image, max_area=100_000)

    W_, H_ = controlled_image.size
    print(f'\n Original size: {W} x {H} = {W*H}')
    print(f'\n 8-Divided size: {W_} x {H_} = {W_*H_}')

    ## Diffusion-based Generation
    from .pipeline import run_pipeline

    diffusion_kwargs = dict(

        # Prompt params
                 prompt = prompt,
        negative_prompt = nrompt,

        # Common diffusion params
        guidance_scale = float(guidance_scale), 
        num_inference_steps = num_steps, 
        height = H_,
        width = W_,

        # Controlnet params
        conditioning_frames = controlled_image,
        controlnet_frame_indices = [0],
        controlnet_conditioning_scale = float(controlnet_scale),

        # Video params
        num_frames = num_frames,
    )

    video = run_pipeline(sd_version, ckpt_path, vae_path, 
                                     lora_path, adapter_path, ctrlnet_path,
                                    **diffusion_kwargs)

    ## Postprocessing
    ## - Naïve: https://gist.github.com/shivasiddharth/3ee632ce6513bc6ae956f58476983659
    ## - DNN: https://github.com/daQuincy/opencv-superres-react/blob/main/super_res_video.py
    # video = upscale(video)

    return video


## Define UI settings & layout 

def create_ui(
    models_path=None, model_default=None,
    # loras_path=None, lora_default=None,
    adapters_path=None, adapter_default=None,
    # controlnets_path=None, controlnet_default=None,
    min_width: int = 25,
):

    global ckpt_lookup, vae_lookup, adapter_lookup, lora_lookup, ctrlnet_lookup

    if models_path is not None:
        print('Overwrite SD checkpoints lookup-table ...')
        ckpt_lookup = models_path

    if adapters_path is not None:
        print('Overwrite Adapters lookup-table ...')
        adapter_lookup = adapters_path

    # for ckpt_name, ckpt_path in ckpt_lookup.items():
    #     if ckpt_path.endswith(tuple(MODEL_EXTENSIONS)):
    #         del ckpt_lookup[ckpt_name]

    MODELS = list(ckpt_lookup.keys())
    ENCODERS = list(vae_lookup.keys())
    ADAPTERS = list(adapter_lookup.keys())
    CTRLNETS = list(ctrlnet_lookup.keys())
    LORAS = list(lora_lookup.keys())

    column_kwargs = dict(variant='panel', min_width=min_width)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:
        
        gr.Markdown("## ୧⍤⃝ Animation")
        
        with gr.Row():
            prompt_display_config = dict(max_lines=7, lines=3)
            with gr.Column(scale=2, **column_kwargs) as prompt_panel:
                prompt = gr.Textbox(label='Positive Prompt', value=POSITIVE_PROMPT, placeholder="Prompt to generate animation", **prompt_display_config)
            with gr.Column(scale=1, **column_kwargs) as prompt_panel:
                nrompt = gr.Textbox(label='Negative Prompt', value=NEGATIVE_PROMPT, placeholder="Prompt to exclude in animation", **prompt_display_config)

        with gr.Row() as ckpt_panel:
            with gr.Column(scale=1, **column_kwargs):
                modelname = gr.Dropdown(label='Checkpoint', choices=MODELS, multiselect=False, value=default_model)
                modelclss = gr.Dropdown(label='Version', choices=SD_VERSION, multiselect=False, value=default_sd)
                autoencoder = gr.Dropdown(label='VAEncoder', choices=ENCODERS, multiselect=False, value=default_vae)
            with gr.Column(scale=1, **column_kwargs):
                loradpt = gr.Dropdown(label='LoRAdapter', choices=LORAS, multiselect=False, value=default_lora)
                adapter = gr.Dropdown(label='Ip-Adapter', choices=ADAPTERS, multiselect=False, value=default_adapter)
                ctrlnet = gr.Dropdown(label='ControlNet', choices=CTRLNETS, multiselect=False, value=default_ctrlnet)

        with gr.Row():

            with gr.Column(scale=3, **column_kwargs) as image_panel:
                image = gr.Image(label='Image')

            with gr.Column(scale=2, **column_kwargs) as param_panel:
                guidance = gr.Slider(minimum=1., maximum=49, step=0.1, value=7.7, label='Guidance Scale')
                n_isteps = gr.Slider(minimum=10, maximum=100, step=1, value=10, label='Inference Steps')

                strength = gr.Slider(minimum=.1, maximum=1.99, step=.1, value=1, label='Controlnet Scale')
                n_frames = gr.Slider(minimum=10, maximum=100, step=1, value=10, label='Frames')

                button = gr.Button(value="Generate", variant='primary')

            with gr.Column(scale=3, **column_kwargs) as output_panel:
                video = gr.Image(label='Video')

        ## Functionality
        models = [modelname, modelclss, autoencoder, loradpt, adapter, ctrlnet]
        params = [guidance, n_isteps, strength, n_frames]

        button.click(fn=generate_video, inputs=[prompt, nrompt, image]+models+params, outputs=[video])
    
    return gui

