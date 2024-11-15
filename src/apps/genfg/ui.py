"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import os 
from copy import deepcopy

import gradio as gr
import numpy as np

from PIL import Image, ImageOps
from PIL.Image import Image as PILImage

from src.config import sd15_ckpt_dirs, lora_dirs
from src.utils import scan_checkpoint_dir, prettify_dict
from src.utils import preprocess_image, validate_gpu_memory
from src.utils import POSITIVE_PROMPT, NEGATIVE_PROMPT
from .examples import TEXT_EXAMPLES, IMAGE_EXAMPLES
from .styles import STYLES_PROMPT, STYLES


## Variables

SD_VERSION = ['SD-15']
default_sd = 'SD-15'


ckpt_lookup = dict()
sd_ckpt_dirs = sd15_ckpt_dirs
for ckpt_dir in sd_ckpt_dirs:
    ckpt_dict = scan_checkpoint_dir(ckpt_dir)
    ckpt_lookup.update(ckpt_dict)

default_model = 'dreamshaper_8_inpainting'


lora_lookup = dict()
for lora_dir in lora_dirs:
    lora_dict = scan_checkpoint_dir(lora_dir)
    lora_lookup.update(lora_dict)

default_lora = None
# prettify_dict(lora_lookup)


## Pipeline

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
        adapter, adapter_strength, adapter_mode,
        ckpt, sd_version, model_channel,
        image, mask, 
        strength, guidance, num_steps, batch_size,
    ):
    
    ckpt_path = ckpt_lookup[ckpt]
    adapter_path = adapter_lookup[adapter]
    iencoder_path = image_encodirs[adapter]

    ## Preprocessing
    (image, mask), (W, H) = preprocess_image(image, mask, max_area=250_000)

    ## Diffusion-based Generation
    from .pipelines.transfer import run_pipeline

    diffusion_kwargs = dict( batch_size = batch_size, 
                               strength = strength, 
                          adapter_scale = adapter_strength,
                         guidance_scale = guidance, 
                    num_inference_steps = num_steps, )

    generated = run_pipeline(ckpt_path, adapter_path, iencoder_path, 
                            adapter_mode, sd_version, model_channel,
                             image, mask, styled_image, **diffusion_kwargs)

    ## Output Formatting --> Gallery
    outputs = [(img.resize(size=(W, H)), f'gen_{i}') for i, img in enumerate(generated)]

    return outputs


## Define UI settings & layout 

def create_ui(
    object_image=None, mask_image=None,
    models_path=None, model_default=None,
    adapters_path=None, adapter_default=None,
    min_width: int = 25, 
    **kwargs
):
    
    global ckpt_lookup, adapter_lookup

    if models_path is not None:
        print('Overwrite SD checkpoints lookup-table ...')
        ckpt_lookup = models_path

    if adapters_path is not None:
        print('Overwrite Adapters lookup-table ...')
        adapter_lookup = adapters_path

    MODELS = list(ckpt_lookup.keys())
    ADAPTERS = list(adapter_lookup.keys())

    column_kwargs = dict(variant='panel', min_width=min_width)

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
                img_style = gr.Image(label='Style Image')
                adapter = gr.Dropdown(label='Ip-Adapter', choices=ADAPTERS, value=default_adapter, multiselect=False)
                adamode = gr.Dropdown(label='Ip-Mode', choices=adapter_modes, value='style only', multiselect=False)
                adascale = gr.Slider(label='Ip-Strength', minimum=.1, maximum=1.99, step=.01, value=0.55)
                img_gen = gr.Button(value="Generate", variant='primary')

            with gr.Column(scale=2, **column_kwargs) as ckpt_panel:
               
                with gr.Row():
                    modelname = gr.Dropdown(label='Checkpoint', choices=MODELS, multiselect=False, value=default_model)
                    modelclss = gr.Dropdown(label='Version', choices=SD_VERSION, multiselect=False, value=default_sd)
                    modelchnl = gr.Dropdown(label='In Channels', choices=[4, 9], multiselect=False, value=9)
                
                strength = gr.Slider(minimum=.1, maximum=.99, step=.01, value=0.9, label='Strength')
                guidance = gr.Slider(minimum=1., maximum=49, step=0.1, value=16.9, label='Guidance Scale')

            with gr.Column(scale=3, **column_kwargs) as output_panel:
                
                # genlery = gr.Image(label="Generated images")
                genlery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery", 
                                    columns=[4], rows=[1], object_fit="contain", height="auto")
                with gr.Row():
                    batch_size = gr.Slider(minimum=1, maximum=100, step=1, value=1, label='Batch Size')
                    num_steps = gr.Slider(minimum=25, maximum=100, step=1, value=50, label='Diffusion Steps')
            
        with gr.Row(visible=True) as text_recmmd:
            examples, nameples = [], []
            for e in TEXT_EXAMPLES:
                examples.append(e[1:])
                nameples.append(e[0])
            texamples = gr.Examples(
                            example_labels=nameples,
                            examples=examples,
                            inputs=[prompt, modelname, modelclss, modelchnl, 
                                    strength, guidance, num_steps])

        with gr.Row(visible=False) as image_recmmd:
            examples, nameples = [], []
            for e in IMAGE_EXAMPLES:
                examples.append(e[1:])
                nameples.append(e[0])
            iexamples = gr.Examples(
                            example_labels=nameples,
                            examples=examples,
                            inputs=[img_style, adascale, 
                                    modelname, modelclss, modelchnl, strength, guidance, num_steps])

        ## Functionality
        shared_inputs = [modelname, modelclss, modelchnl,
                         object_image, mask_image, 
                         strength, guidance, num_steps, batch_size]

        txt_gen.click(fn=generate_by_text, inputs=[prompt, styles]+shared_inputs, outputs=[genlery])
        img_gen.click(fn=generate_by_image, inputs=[img_style, 
                                        adapter, adascale, adamode]+shared_inputs, outputs=[genlery])

        switch_modality = lambda x: [gr.update(visible = x), gr.update(visible = not x), 
                                     gr.update(visible = x), gr.update(visible = not x)]
        
        prompt_modality.change(fn=switch_modality, inputs=[prompt_modality], 
                                                  outputs=[prompt_image, prompt_text, 
                                                           image_recmmd, text_recmmd])
    return gui, [genlery]


if __name__ == "__main__":

    gui, *_ = create_ui()
    gui.launch(server_name='localhost', server_port=7861, share=False)

