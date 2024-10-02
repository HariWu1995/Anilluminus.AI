import gradio as gr


# Define styles
min_width = 25
main_theme = gr.themes.Soft(primary_hue=gr.themes.colors.blue, 
                          secondary_hue=gr.themes.colors.sky,)

css = """
.gradio-container {width: 95% !important}
"""

# Define texts
title = r"""
<h1 align="center">Anilluminus.AI</h1>
"""

description = r"""
<b>Gradio demo</b> for <a href='https://github.com/HariWu1995/Anilluminus.AI' target='_blank'><b> Anilluminus.AI </b></a>.<br>
"""