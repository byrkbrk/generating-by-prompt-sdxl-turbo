from argparse import ArgumentParser
from stable_diffusion_xl_turbo import StableDiffusion



def parse_arguments():
    """Returns parsed arguments"""
    parser = ArgumentParser(description="Generate by prompt using SDXL-turbo")
    parser.add_argument("prompt", nargs="+", type=str, help="Text prompts that be used for generating")
    parser.add_argument("--num_inference_steps", type=int, default=5, 
                        help="Number of inferences steps that be used during inference. Default: 5")
    parser.add_argument("--device", type=str, default=None, help="Device name used for inferece. Default: None")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    StableDiffusion(device=args.device).generate(args.prompt, args.num_inference_steps)
