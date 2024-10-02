"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import os
import gradio as gr
import numpy as np
import cv2

from io import BytesIO
from typing import List, Union

from PIL import Image, ImageOps
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

    mask = alpha.point(lambda p: 255 - p)

    # obj = Image.merge("RGB", rgb)
    # obj.putalpha(mask.convert('L'))

    return img, mask


def invert_mask(mask):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask = ImageOps.invert(mask)
    return mask


def apply_mask(image, mask):
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask).convert('L')
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    image.putalpha(mask)
    return image


def expand_img(image, left: int = 0, right: int = 0, 
                       top: int = 0, bottom: int = 0):

    paddings = (top, bottom, left, right)
    
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_REPLICATE)
    return image


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

            with gr.Column(scale=2, **column_kwargs):
                run_button = gr.Button(value="Decompose", variant='primary')
                inv_button = gr.Button(value="Invert Mask", variant='secondary')
                # app_button = gr.Button(value="Apply Mask", variant='secondary')
                # send_button = gr.Button(value="Send ⇩", variant='secondary')
                exp_button = gr.Button(value="Expand", variant='secondary')

            with gr.Column(scale=4, **column_kwargs):

                model = gr.Dropdown(label="Background Remover", choices=models, value="u2net")
                alpha = gr.Checkbox(label="Alpha matting", value=False)
                # mask = gr.Checkbox(label="Return mask", value=False)
                expnd = gr.Checkbox(label="Expansion", value=False)

            with gr.Column(scale=4, visible=False, **column_kwargs) as alpha_mask_options:
                alpha_erosion_size = gr.Slider(label="Erosion size"        , minimum=0, maximum= 40, step=1, value= 10)
                alpha_fg_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240)
                alpha_bg_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value= 10)
                
            with gr.Column(scale=1, visible=False, **column_kwargs) as expansion_options:
                l_size = gr.Slider(label="Left"  , minimum=0, maximum=255, step=1, value=0)
                r_size = gr.Slider(label="Right" , minimum=0, maximum=255, step=1, value=0)
                t_size = gr.Slider(label="Top"   , minimum=0, maximum=255, step=1, value=0)
                b_size = gr.Slider(label="Bottom", minimum=0, maximum=255, step=1, value=0)

        ## Functionality
        rembg_inputs = [img_in, alpha, alpha_fg_threshold, alpha_bg_threshold, alpha_erosion_size, model]
        display_block = lambda x: gr.update(visible=x)

        alpha.change(fn=display_block, inputs=[alpha], outputs=[alpha_mask_options])
        expnd.change(fn=display_block, inputs=[expnd], outputs=[expansion_options])

        # app_button.click(fn=apply_mask, inputs=[img_out, img_mask], outputs=[img_mask])
        exp_button.click(fn=expand_img, inputs=[img_in, 
                                                l_size, r_size, 
                                                t_size, b_size], outputs=[img_in])

        inv_button.click(fn=invert_mask, inputs=[img_mask], outputs=[img_mask])
        run_button.click(fn=run_rembg, inputs=rembg_inputs, outputs=[img_out, img_mask])

    return gui, [img_out, img_mask]

