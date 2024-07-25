# Generating by prompts using SDXL-turbo

## Introduction

We implement a module that generates images based on user-defined (text) prompts. We use the pretrained [Stable Diffusion XL-Turbo](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) provided by *stabilityai* at HuggingFace, while preparing the module.

## Setting Up the Environment


## Generating Images

### Example usage

~~~
python3 generate.py\
 "an image of turtle in Picasso style"\
 "an image of turtle in Camille Pissarro style"\
 --num_inference_steps 10
~~~

The output images seen below (left: Picasso style, right: Pissarro style) will be saved into `generated-images` folder.

<p align="center">
  <img src="files-for-readme/picasso_turtle.png" width="49%" />
  <img src="files-for-readme/pissarro_turtle.png" width="49%" />
</p>



