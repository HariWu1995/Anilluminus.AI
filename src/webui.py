import sys
sys.path.append('../')

import gradio as gr


# Global Variables
from .config import css, title, description, min_width, main_theme
from .usage import tips


# Layout

def load_mini_apps():

    from app.builders.context.ui import create_ui as create_ui_context
    from app.builders.dialogue.ui import create_ui as create_ui_dialogue
    from app.builders.character.ui import create_ui as create_ui_character

    gui_character, in_character, out_character = create_ui_character()
    gui_dialogue,  in_dialogue                 = create_ui_dialogue()
    gui_context,   in_context,  out_context    = create_ui_context()

    return  (gui_context, gui_character, gui_dialogue), \
            ( in_context,  in_character,  in_dialogue), \
            (out_context, out_character,             )


def load_shared_story(size: str = 'sm', variant: str = 'secondary'):

    style_kwargs = dict(size=size, variant=variant)

    with gr.Row():
        with gr.Column(scale=1, variant='panel', min_width=min_width):
            button_mas2all  = gr.Button(value="Send to All")
        with gr.Column(scale=1, variant='panel', min_width=min_width):
            button_mas2char = gr.Button(value="Send to Character Builder", **style_kwargs)
        with gr.Column(scale=1, variant='panel', min_width=min_width):
            button_mas2ctx  = gr.Button(value="Send to Context Builder", **style_kwargs)
        with gr.Column(scale=2, variant='panel', min_width=min_width):
            gr.Markdown('')
    
    return button_mas2all, button_mas2char, button_mas2ctx


def run_demo(server: str = 'localhost', port: int = 7861, share: bool = False):

    from app.builders.story.ui import create_ui as create_ui_mastory

    tabs, (in_ctx, in_char, in_chat), \
         (out_ctx, out_char,       ) = load_mini_apps()

    # Handle shared data between mini-apps
    char_1_name, char_1_core, char_1_mem, \
    char_2_name, char_2_core, char_2_mem, \
                                fw_char_button = out_char
    time_context, place_context, fw_ctx_button = out_ctx
    time_dialog, place_dialog, \
                        chnm_1, char_1, chev_1, \
                        chnm_2, char_2, chev_2 = in_chat

    names = ["Context Builder", "Character Builder", "Dialogue"]

    with gr.Blocks(css=css, theme=main_theme, analytics_enabled=False) as demo:
        
        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        # Body
        master_ui, mastory = create_ui_mastory(all_themes)

        transfer_data = lambda x: x
        transdup_data = lambda x: [x, x]
        transfer_list = lambda x, y: [x, y]
        transfx6_list = lambda x1, x2, x3, x4, x5, x6: [x1, x2, x3, x4, x5, x6]
                
        button_mas2all, button_mas2chr, button_mas2ctx = load_shared_story()

        button_mas2all.click(fn=transdup_data, inputs=mastory, outputs=[in_char, in_ctx])
        button_mas2chr.click(fn=transfer_data, inputs=mastory, outputs=in_char)
        button_mas2ctx.click(fn=transfer_data, inputs=mastory, outputs=in_ctx)

        fw_char_button.click(fn=transfx6_list, inputs=[char_1_name, char_1_core, char_1_mem, \
                                                       char_2_name, char_2_core, char_2_mem], 
                                              outputs=[chnm_1,      char_1,      chev_1,     
                                                       chnm_2,      char_2,      chev_2    ])
        fw_ctx_button.click(fn=transfer_list, inputs=[time_context, place_context], 
                                             outputs=[time_dialog, place_dialog])

        gr.TabbedInterface(interface_list=tabs, tab_names=names)

        # Footer
        gr.Markdown(tips)

    demo.launch(server_name=server, server_port=port, share=share)


if __name__ == "__main__":

    run_demo(server='localhost', port=7861, share=False)

