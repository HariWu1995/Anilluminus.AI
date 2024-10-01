"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import os
import gradio as gr
import numpy as np

from io import BytesIO
from typing import List

from PIL import Image
from PIL.Image import Image as ImageClass

from .utils import remove_background
from .models import sessions_names as models
from .session import new_session


default_model = 'u2net'
all_models = [
    "u2net","u2netp","u2net_human","u2net_cloth","silueta",
    "isnet-general","isnet-anime",
]


def run_rembg(img, alpha, alpha_fg_threshold, alpha_bg_threshold, alpha_erosion_size, model):
    
    masked_obj = remove_background(img, alpha, 
                                        alpha_fg_threshold, 
                                        alpha_bg_threshold, 
                                        alpha_erosion_size, model)
    # Split layers
    masked_obj = masked_obj.convert("RGBA")
    *rgb, alpha = masked_obj.split()

    obj = Image.merge("RGB", rgb)
    mask = alpha

    return obj, mask


# Define UI settings & layout

def create_ui(min_width: int = 25):

    column_kwargs = dict(variant='panel', min_width=min_width)
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🖼 Background Decomposition")

        with gr.Row():
            img_in = gr.Image(label='Input')
            img_out = gr.Image(label='Object')
            img_mask = gr.Image(label='Mask')

        with gr.Row():

            with gr.Column(scale=1, **column_kwargs):
                run_button = gr.Button(value="Decompose", variant='primary')
                # send_button = gr.Button(value="Send ⇩", variant='secondary')

            with gr.Column(scale=2, **column_kwargs):
                model = gr.Dropdown(label="Remove background", choices=models, value="u2net")
                alpha = gr.Checkbox(label="Alpha matting", value=False)
                # mask = gr.Checkbox(label="Return mask", value=False)

            with gr.Column(scale=2, visible=False, **column_kwargs) as alpha_mask_options:
                alpha_erosion_size = gr.Slider(label="Erosion size"        , minimum=0, maximum= 40, step=1, value= 10)
                alpha_fg_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240)
                alpha_bg_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value= 10)

        ## Functionality
        rembg_inputs = [img_in, alpha, alpha_fg_threshold, alpha_bg_threshold, alpha_erosion_size, model]
        display_alpha = lambda x: gr.update(visible=x)

        alpha.change(fn=display_alpha, inputs=[alpha], outputs=[alpha_mask_options])
        run_button.click(fn=run_rembg, inputs=rembg_inputs, outputs=[img_out, img_mask])

    return gui, [img_out, img_mask]

