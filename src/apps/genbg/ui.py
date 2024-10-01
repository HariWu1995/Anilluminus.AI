"""
Emojis: 🎑🌁🌆🖼️🏞️🌄🏜️🏙🗺️🖼🌉🏕️
"""
import os 
import gradio as gr

from .models import *
from .session import new_session


all_models = sessions_names
default_model = 'u2net'


# Define UI settings & layout

def create_ui(min_width: int = 25):
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        ## Background Removal
        gr.Markdown("## 🖼 Background Removal")

        with gr.Row():
                model = gr.Dropdown(label="Remove background", choices=models, value="None")
                return_mask = gr.Checkbox(label="Return mask", value=False)
                alpha_matting = gr.Checkbox(label="Alpha matting", value=False)



        ## Background 
        gr.Markdown("## 🏙 Background Stylization")



