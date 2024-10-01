import sys
sys.path.append('../')

import gradio as gr


# Global Variables
from .config import css, title, description, min_width, main_theme
from .usage import tips


# Layout

def load_mini_apps():

    from src.apps.rembg.ui import create_ui as create_ui_rembg

    gui_rembg, *out_rembg = create_ui_rembg()

    return  (gui_rembg, ), \
            (out_rembg, )


def run_demo(server: str = 'localhost', port: int = 7861, share: bool = False):

    tabs, (out_rembg) = load_mini_apps()

    names = ["Background"]

    with gr.Blocks(css=css, theme=main_theme, analytics_enabled=False) as demo:
        
        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        # Body

        transfer_data = lambda x: x
        transdup_data = lambda x: [x, x]
        transfer_list = lambda x, y: [x, y]

        gr.TabbedInterface(interface_list=tabs, tab_names=names)

        # Footer
        gr.Markdown(tips)

    demo.launch(server_name=server, server_port=port, share=share)


if __name__ == "__main__":

    run_demo(server='localhost', port=7861, share=False)

