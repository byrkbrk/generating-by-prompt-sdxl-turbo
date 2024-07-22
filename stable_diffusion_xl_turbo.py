import os
import torch
from diffusers import DiffusionPipeline



class StableDiffusion(object):
    def __init__(self, 
                 model_id="stabilityai/sdxl-turbo",
                 device="mps",
                 ):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device(device)
        self.pipeline = self.instantiate_pipeline(model_id, self.device)
        self.create_dirs(self.module_dir)

    def generate(self, prompt, num_inference_steps=5, show=True, save=True):
        """Returns generated image for given prompt"""
        images = self.pipeline(prompt, num_inference_steps=num_inference_steps).images
        for i, image in enumerate(images):
            if save:
                image.save(os.path.join(self.module_dir, "generated-images", f"generated_image_{i}.png"))
            if show:
                image.show()
        return images

    def instantiate_pipeline(self, model_id, device):
        """Returns instantiated pipeline"""
        pipeline = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            ).to(device)
        return pipeline
    
    def initialize_device(self, device):
        """Returns the initialized device based on GPU availability"""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def create_dirs(self, root):
        """Creates required directories for inference"""
        dir_names = ["generated-images"]
        for dir_name in dir_names:
            os.makedirs(os.path.join(root, dir_name), exist_ok=True)



if __name__ == "__main__":
    stable_diffusion = StableDiffusion()
    prompt = ["an image of a turtle in Camille Pissarro style", "an image of a turtle in Picasso style"]
    stable_diffusion.generate(prompt, num_inference_steps=15)


