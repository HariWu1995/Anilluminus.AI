"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import os 
from copy import deepcopy

import gradio as gr
import numpy as np

from PIL import Image, ImageOps
from PIL.Image import Image as PILImage

from src.utils import scan_checkpoint_dir, find_divisible, \
                        POSITIVE_PROMPT, NEGATIVE_PROMPT
from .styles import STYLES, STYLES_PROMPT
from .examples import TEXT_EXAMPLES, IMAGE_EXAMPLES


SD_VERSION = ['SD-15','SD-XL','BrushNet','BrushNet-XL']

ckpt_dirs = ['./checkpoints/models']
ckpt_lookup = dict()
for ckpt_dir in ckpt_dirs:
    ckpt_dict = scan_checkpoint_dir(ckpt_dir)
    ckpt_lookup.update(ckpt_dict)

default_model = 'dreamshaper_inpainting_v8'


def preprocess_image(image, mask=None):

    def find_divisible_by_8(*X, return_mode: str = 'nearest'):
        return [find_divisible(x, frac=8, return_mode=return_mode) for x in X]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    elif not isinstance(image, PILImage):
        raise TypeError(f"{image.__class__} is not supported!")

    image = image.convert('RGB')
    W, H = image.size

    ## Auto-Scale
    W_new, H_new = find_divisible_by_8(W, H, return_mode='nearest')
    while (H_new * W_new) > 500_000:
        W_new = int(W_new * 0.69)
        H_new = int(H_new * 0.69)
        W_new, H_new = find_divisible_by_8(W_new, H_new, return_mode='lower')

    image = image.resize((W_new, H_new))

    if mask is None:
        return image, (W, H)

    mask = deepcopy(mask)

    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)

    elif not isinstance(mask, PILImage):
        raise TypeError(f"{mask.__class__} is not supported!")

    mask = mask.convert('L').resize((W_new, H_new))
    return (image, mask), (W, H)


# Pipeline

def generate_by_text(
        prompt, styles, 
        ckpt, sd_version, model_channel, 
        image, mask, 
        strength, guidance, num_steps, batch_size,
    ):

    ckpt_path = ckpt_lookup[ckpt]

    ## Preprocessing
    (image, mask), (W, H) = preprocess_image(image, mask)

    ## Prompt Engineering
    if prompt == '':
        prompt = POSITIVE_PROMPT
        if styles is not None:
            if styles != '':
                prompt += (' '.join([STYLES_PROMPT[s] for s in styles]))

    else:
        if styles is not None:
            if styles != '':
                prompt += ', '.join(styles)
        prompt += ('. ' + POSITIVE_PROMPT)

    ## Diffusion-based Generation
    from .pipelines.outpaint import run_pipeline

    diffusion_kwargs = dict( batch_size = batch_size, 
                               strength = strength, 
                         guidance_scale = guidance, 
                    num_inference_steps = num_steps, output_type = 'pil', )

    generated = run_pipeline(ckpt_path, sd_version, model_channel,
                             image, mask, prompt, NEGATIVE_PROMPT, **diffusion_kwargs)

    ## Output Formatting --> Gallery
    outputs = [(img.resize(size=(W, H)), f'gen_{i}') for i, img in enumerate(generated)]

    return outputs


def generate_by_image(
        styled_image, 
        ckpt, sd_version, model_channel,
        image, mask, 
        strength, guidance, num_steps, batch_size,
    ):

    ckpt_path = ckpt_lookup[ckpt]

    return [(image, 'gen_001'),
            (image, 'gen_002'),
            (image, 'gen_003'),
            (image, 'gen_004'),]


# Define UI settings & layout 

def create_ui(
    object_image=None, mask_image=None, 
    models_path=None, model_default=None,              
    min_width: int = 25
):
    
    column_kwargs = dict(variant='panel', min_width=min_width)

    global ckpt_lookup

    if models_path is not None:
        print('Overwrite SD checkpoints lookup-table ...')
        ckpt_lookup = models_path

    MODELS = list(ckpt_lookup.keys())

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🏙 Background Stylization")

        if (object_image is None) or (mask_image is None):
            with gr.Row():
                object_image = gr.Image(label='Object')
                mask_image = gr.Image(label='Mask')

        prompt_modality = gr.Checkbox(label="Image Style-Transfer", value=False)

        with gr.Row():

            with gr.Column(scale=3, visible=True, **column_kwargs) as prompt_text:
                prompt = gr.Textbox(label='Text Prompt', max_lines=7, lines=3, placeholder="Prompt to generate style")
                styles = gr.Dropdown(label='Style Prompt', choices=STYLES, multiselect=True, max_choices=3)
                txt_gen = gr.Button(value="Generate", variant='primary')

            with gr.Column(scale=3, visible=False, **column_kwargs) as prompt_image:
                image_styled = gr.Image(label='Style Image')
                img_gen = gr.Button(value="Generate", variant='primary')

            with gr.Column(scale=2, **column_kwargs):
               
                with gr.Row():
                    modelname = gr.Dropdown(label='Checkpoint', choices=MODELS, multiselect=False, value=default_model)
                    modelclss = gr.Dropdown(label='Version', choices=SD_VERSION, multiselect=False, value='SD-15')
                    modelchnl = gr.Dropdown(label='In Channels', choices=[4, 9], multiselect=False, value=9)
                
                strength = gr.Slider(minimum=.1, maximum=.99, step=.01, value=0.9, label='Strength')
                guidance = gr.Slider(minimum=1., maximum=49, step=0.1, value=16.9, label='Guidance Scale')

            with gr.Column(scale=3, **column_kwargs):
                
                with gr.Row():
                    batch_size = gr.Slider(minimum=1, maximum=100, step=1, value=1, label='Batch Size')
                    num_steps = gr.Slider(minimum=25, maximum=100, step=1, value=50, label='Inference Steps')
                
                # genlery = gr.Image(label="Generated images")
                genlery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery", 
                                    columns=[4], rows=[1], object_fit="contain", height="auto")
            
        with gr.Row():
            examples, nameples = [], []
            for e in TEXT_EXAMPLES:
                examples.append(e[1:])
                nameples.append(e[0])
            texamples = gr.Examples(
                            example_labels=nameples,
                            examples=examples,
                            inputs=[prompt, modelname, modelclss, modelchnl, 
                                    strength, guidance, num_steps])

        ## Functionality
        shared_inputs = [modelname, modelclss, modelchnl,
                         object_image, mask_image, 
                         strength, guidance, num_steps, batch_size]

        txt_gen.click(fn=generate_by_text, inputs=[prompt, styles]+shared_inputs, outputs=[genlery])
        img_gen.click(fn=generate_by_image, inputs=[image_styled]+shared_inputs, outputs=[genlery])

        switch_modality = lambda x: [gr.update(visible = x), gr.update(visible = not x)]
        prompt_modality.change(fn=switch_modality, inputs=[prompt_modality], 
                                                  outputs=[prompt_image, prompt_text])

    return gui, [genlery]





