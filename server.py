import glob
import os
from typing import List

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import fontogen_config
from sampler import create_sampler

# Import your sampler here. For instance:
# from your_module import sampler

FONT_DIR = "./training/samples/"
app = FastAPI()
sampler = None
config = fontogen_config()

CHECKPOINT_DIR = "."
available_checkpoints = sorted(
    [os.path.relpath(f, CHECKPOINT_DIR) for f in glob.glob(f"{CHECKPOINT_DIR}/**/*.ckpt", recursive=True)])
curr_checkpoint_path = None


def generate_font_styles(fonts):
    styles = []
    for font in fonts:
        name = font.split('.')[0]
        style_name = "FontoGen-" + name
        style = f"""
        @font-face {{
            font-family: '{style_name}';
            src: url(/fonts/{font});
        }}
        .{style_name.replace(",", "_").replace(";", "_")} {{
            font-family: '{style_name}', serif;
            font-size: 32px;
            width: 100%;
        }}
        """
        styles.append(style)
    return "\n".join(styles)


def generate_font_paragraphs(glyphs: str, fonts: List[str]):
    paragraphs = []
    for font in fonts:
        name = font.split('.')[0]
        style_name = "FontoGen-" + name
        paragraph = f"""
        <p>
            {name}
        </p>
        <p class="{style_name.replace(",", "_").replace(";", "_")}">
            {glyphs}
        </p>
        """
        paragraphs.append(paragraph)
    return "\n".join(paragraphs)


def font_generator(font_txt, temperature: int = None, glyphs: str = None, strategy: str = None,
                   checkpoint_path: str = None):
    # If a description is provided, generate a new font using the sampler
    if font_txt != '$all':
        # Call your sampler here.
        global sampler
        global curr_checkpoint_path
        if sampler is None or sampler.glyphs != glyphs or curr_checkpoint_path != checkpoint_path:
            sampler = create_sampler(FONT_DIR, glyphs=glyphs, checkpoint_path=checkpoint_path)
            curr_checkpoint_path = checkpoint_path
        font_path = sampler.sample(font_txt, -1, temperature=temperature, strategy=strategy)
        fonts = [os.path.basename(font_path)]
    else:
        fonts = sorted([f for f in os.listdir(FONT_DIR) if f.endswith('.ttf')],
                       key=lambda x: os.path.getctime(os.path.join(FONT_DIR, x)),
                       reverse=True)
    glyphs = config.glyphs

    # Return the styles and paragraphs (updated if a new font was generated)
    return f"""
    <style>
    {generate_font_styles(fonts)}
    </style>
    {generate_font_paragraphs(glyphs, fonts)}
    """


with gr.Blocks() as demo:
    title = gr.Markdown(
        f"<h1 style='text-align: center; margin-bottom: 1rem'>FontoGen</h1>"
    )
    model_dropdown = gr.Dropdown(label='Select Model Checkpoint', choices=available_checkpoints)
    custom_glyphs_input = gr.Textbox(label="Glyphs Set", value=config.glyphs)
    name = gr.Textbox(label="Font Style")
    with gr.Row():
        temperature_slider = gr.Slider(minimum=0, maximum=3.0, value=0.6, label='Temperature', interactive=True)
        sampling_strategy_dropdown = gr.Dropdown(
            label='Sampling Strategy',
            value='multinomial',
            interactive=True,
            choices=[
                'multinomial',
                'greedy',
                'topknuc',
            ])
    gen_button = gr.Button("Generate", variant='primary')
    refresh_button = gr.Button("Refresh Fonts", variant='secondary')  # New refresh button
    result = gr.HTML()
    all_fonts = gr.HTML(value=font_generator('$all'))


    @gen_button.click(inputs=[name, model_dropdown], outputs=[result])  # Update inputs here
    def generate(font_text, selected_model):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, selected_model)
        gen_font_result = font_generator(
            font_text,
            temperature=temperature_slider.value,
            glyphs=custom_glyphs_input.value,
            checkpoint_path=checkpoint_path
        )
        return gen_font_result


    @refresh_button.click(outputs=[all_fonts])
    def refresh():
        return font_generator('$all')

app.mount("/fonts/", StaticFiles(directory='./samples/'), name="fonts")
app.mount("/", demo.app)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8888)
