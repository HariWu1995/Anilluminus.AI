"""
!!! Use OS.environ inside script not working

For low-memory GPU:
    Windows: 
        set 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:25'
        set 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
    Linux: export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:25'

!!! CachingAllocator option max_split_size_mb too small, must be > 20
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import sys
sys.path.append('../')

import gradio as gr


## Global Variables
from .config import sd_ckpt_dirs, adapter_dirs
from .layout import css, title, description, min_width, main_theme
from  .usage import tips
from src.utils import scan_checkpoint_dir, prettify_dict

ckpt_lookup = dict()
for ckpt_dir in sd_ckpt_dirs:
    ckpt_dict = scan_checkpoint_dir(ckpt_dir)
    ckpt_lookup.update(ckpt_dict)

adapter_lookup = dict()
for adapter_dir in adapter_dirs:
    adapter_dict = scan_checkpoint_dir(adapter_dir, sub_model=True)
    adapter_lookup.update(adapter_dict)

# prettify_dict(ckpt_lookup)
# prettify_dict(adapter_lookup)


## Layout

transfer_data = lambda x: x
transdup_data = lambda x: [x, x]
transfer_list = lambda x, y: [x, y]


def load_mini_apps():

    global ckpt_lookup

    from src.apps.rembg.ui import create_ui as create_ui_rembg
    from src.apps.genbg.ui import create_ui as create_ui_genbg
    from src.apps.anime.ui import create_ui as create_ui_anime

    with gr.Blocks(css=None, analytics_enabled=False) as gui_background:

        gui_rembg, img_rembg = create_ui_rembg()
        gui_genbg, img_genbg = create_ui_genbg(*img_rembg[:2], models_path=ckpt_lookup, 
                                                             adapters_path=adapter_lookup)

    gui_anime = create_ui_anime(models_path=ckpt_lookup, adapters_path=adapter_lookup)

    return (gui_background, gui_anime), (img_genbg[-1], )


def run_demo(server: str = 'localhost', port: int = 7861, share: bool = False):

    tabs, *_ = load_mini_apps()

    names = ["Background", "Animation"]

    with gr.Blocks(css=css, theme=main_theme, analytics_enabled=False) as demo:
        
        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        # Body

        gr.TabbedInterface(interface_list=tabs, tab_names=names)

        # Footer
        gr.Markdown(tips)

    demo.launch(server_name=server, server_port=port, share=share)


if __name__ == "__main__":

    run_demo(server='localhost', port=7861, share=False)

