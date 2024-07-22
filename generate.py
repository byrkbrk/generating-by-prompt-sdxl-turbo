from argparse import ArgumentParser
from stable_diffusion_xl_turbo import StableDiffusion



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate images by prompts using SDXL-turbo")
    parser.add_argument("prompt", nargs="+", type=str, help="Text prompts that be used for generating")
    parser.add_argument("--num_inference_steps", type=int, default=5, 
                        help="Number of inferences steps that be used during inference. Default: 5")
    parser.add_argument("--device", type=str, default=None, help="Device name used for inferece. Default: None")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image. Default: 512")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image. Default: 512")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    StableDiffusion(device=args.device).generate(
        args.prompt, 
        args.num_inference_steps,
        height=args.height,
        width=args.width)
