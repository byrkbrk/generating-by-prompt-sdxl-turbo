import gradio as gr
from stable_diffusion_xl_turbo import StableDiffusion



if __name__ == "__main__":
    stable_diffusion = StableDiffusion(create_dirs=False)
    gr_interface = gr.Interface(
        fn=lambda prompt, inference_steps: stable_diffusion.generate(prompt,
                                                                     inference_steps,
                                                                     show=False,
                                                                     save=False)[0],
        inputs=[gr.Textbox(lines=3, 
                           placeholder="an image of a turtle in Camille Pissarro style",
                           label="Prompt"),
                gr.Slider(minimum=1, maximum=50, step=1, value=5, label="Inference steps")],
        outputs=gr.Image(type="pil"),
        title="Generate by Prompt using SDXL-turbo"
    )
    gr_interface.launch()