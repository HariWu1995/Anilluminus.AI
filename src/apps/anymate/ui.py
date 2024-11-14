import os 
from copy import deepcopy
from PIL import Image

import gradio as gr
import numpy as np

from diffusers.utils import export_to_video

from src.config import sd15_ckpt_dirs, sdxl_ckpt_dirs, vae_dirs, animatediff
from src.utils import scan_checkpoint_dir, validate_gpu_memory, preprocess_image
from src.utils import POSITIVE_PROMPT, NEGATIVE_PROMPT
from .examples import VIDEO_EXAMPLES


## Variables

sd_ckpt_dirs = sd15_ckpt_dirs

ckpt_lookup = dict()
for ckpt_dir in sd_ckpt_dirs:
    ckpt_dict = scan_checkpoint_dir(ckpt_dir)
    ckpt_lookup.update(ckpt_dict)

default_model = 'realisticVisionV51_v20_woVAE'
default_sd = 'SD-15'


vae_lookup = dict()
for vae_dir in vae_dirs:
    vae_dict = scan_checkpoint_dir(vae_dir, sub_model=True)
    vae_lookup.update(vae_dict)

default_vae = 'stabilityai-sd-vae-ft-mse'


## Pipeline

def generate_video(
        ckpt, vae,
        prompt, nrompt, controlled_image,
        guidance_scale, num_steps, 
        controlnet_scale, num_frames, num_seeds,
    ):

    ckpt_path = ckpt_lookup[ckpt]
    vae_path = vae_lookup[vae]

    lora_path    = animatediff['rgb']['lora']
    adapter_path = animatediff['rgb']['adapter']
    ctrlnet_path = animatediff['rgb']['ctrlnet']

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
        conditioning_frames = [controlled_image] * num_seeds,
        controlnet_frame_indices = list(range(num_seeds)),
        controlnet_conditioning_scale = float(controlnet_scale),

        # Video params
        num_frames = num_frames,
        num_videos_per_prompt = 1,
    )

    frames = run_pipeline(default_sd, ckpt_path, vae_path, 
                                     lora_path, adapter_path, ctrlnet_path,
                                    **diffusion_kwargs)
    frames = frames[num_seeds:]

    ## Postprocessing
    video = export_to_video(frames, fps=5, output_video_path='./logs/temp.mp4')

    ## Upscaling
    ## - Naïve: https://gist.github.com/shivasiddharth/3ee632ce6513bc6ae956f58476983659
    ## - DNN: https://github.com/daQuincy/opencv-superres-react/blob/main/super_res_video.py
    # video = upscale(video)

    return video


## Define UI settings & layout 

def create_ui(
    models_path=None, model_default=None,
    vaes_path=None, vae_default=None,
    min_width: int = 25,
):

    global ckpt_lookup, vae_lookup

    if models_path is not None:
        print('Overwrite SD checkpoints lookup-table ...')
        ckpt_lookup = models_path

    if vaes_path is not None:
        print('Overwrite VAE checkpoints lookup-table ...')
        vae_lookup = vaes_path

    MODELS = list(ckpt_lookup.keys())
    ENCODERS = list(vae_lookup.keys())

    column_kwargs = dict(variant='panel', min_width=min_width)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:
                
        with gr.Row() as prompt_panel:
            prompt_display_config = dict(max_lines=7, lines=3)
            with gr.Column(scale=2, **column_kwargs):
                prompt = gr.Textbox(label='Positive Prompt', value=POSITIVE_PROMPT, placeholder="Prompt to describe animation", **prompt_display_config)
            with gr.Column(scale=1, **column_kwargs):
                nrompt = gr.Textbox(label='Negative Prompt', value=NEGATIVE_PROMPT, placeholder="Prompt to exclude in animation", **prompt_display_config)

        with gr.Row() as ckpt_panel:
            modelname = gr.Dropdown(label='Checkpoint', choices=MODELS, multiselect=False, value=default_model)
            autoencoder = gr.Dropdown(label='VAEncoder', choices=ENCODERS, multiselect=False, value=default_vae)

        with gr.Row() as diff_panel:
            guidance = gr.Slider(minimum=1., maximum=19, step=0.1, value=7.7, label='Guidance Scale')
            n_isteps = gr.Slider(minimum=10, maximum=100, step=1, value=49, label='Diffusion Steps')

        with gr.Row():

            with gr.Column(scale=1, **column_kwargs) as ctrl_panel:
                image = gr.Image(label='Image')
                strength = gr.Slider(minimum=.1, maximum=2., step=.05, value=1, label='Control Scale')
                with gr.Row():
                    s_frames = gr.Slider(minimum=1, maximum=10, step=1, value=1, label='Initials')  # duplicate controlled image for more stable
                    n_frames = gr.Slider(minimum=10, maximum=96, step=1, value=10, label='Frames')

            with gr.Column(scale=1, **column_kwargs) as output_panel:
                video = gr.Video(label='Video')
                button = gr.Button(value="Generate", variant='primary')

        models = [modelname, autoencoder]
        inputs = [prompt, nrompt, image]
        params = [guidance, n_isteps, strength, n_frames, s_frames]

        with gr.Row(visible=True) as video_recmmd:
            examples, nameples = [], []
            for e in VIDEO_EXAMPLES:
                examples.append(e[1:])
                nameples.append(e[0])
            vdexamples = gr.Examples(
                            example_labels=nameples,
                            examples=examples,
                            inputs=inputs+models+params)

        ## Functionality

        button.click(fn=generate_video, inputs=models+inputs+params, outputs=[video])
    
    return gui

