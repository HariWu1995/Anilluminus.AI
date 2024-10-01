"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import os 
import gradio as gr

from .styles import STYLES


all_models = []
default_model = 'u2net'


# Pipeline

def generate_by_text(prompt, styles, image, mask):
    return image


def generate_by_image(styled_image, image, mask):
    return image


# Define UI settings & layout 

def create_ui(object_image=None, mask_image=None, min_width: int = 25):
    
    column_kwargs = dict(variant='panel', min_width=min_width)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🏙 Background Stylization")

        if (object_image is None) or (mask_image is None):
            with gr.Row():
                object_image = gr.Image(label='Object')
                mask_image = gr.Image(label='Mask')

        prompt_modality = gr.Checkbox(label="Image Style-Transfer", value=False)

        with gr.Row():

            with gr.Column(scale=1, visible=True, **column_kwargs) as prompt_text:
                prompt = gr.Textbox(label='Text Prompt', max_lines=3, placeholder="Prompt to generate style")
                styles = gr.Dropdown(label='Style Prompt', choices=STYLES, max_choices=3, multiselect=True)
                txt_gen = gr.Button(value="Generate", variant='primary')

            with gr.Column(scale=1, visible=False, **column_kwargs) as prompt_image:
                img_prompt = gr.Image(label='Styple Image')
                img_gen = gr.Button(value="Generate", variant='primary')

            with gr.Column(scale=1, visible=True, **column_kwargs):
                # genlery = gr.Image(label="Generated images")
                genlery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery", 
                                    columns=[4], rows=[1], object_fit="contain", height="auto")

        ## Functionality
        shared_inputs = [object_image, mask_image]
        txt_gen.click(fn=generate_by_text, inputs=[prompt, styles]+shared_inputs, outputs=[genlery])
        img_gen.click(fn=generate_by_image, inputs=[img_prompt]+shared_inputs, outputs=[genlery])

        switch_modality = lambda x: [gr.update(visible = x), gr.update(visible = not x)]
        prompt_modality.change(fn=switch_modality, inputs=[prompt_modality], 
                                                  outputs=[prompt_image, prompt_text])

    return gui, [genlery]





