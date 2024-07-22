import os
import torch
from diffusers import DiffusionPipeline



class StableDiffusion(object):
    def __init__(self, 
                 model_id="stabilityai/sdxl-turbo"
                 ):
        self.module_dir = os.path.dirname(__file__)
        self.device = torch.device("mps") #self.initialize_device()
        self.pipeline = self.instantiate_pipeline(model_id)

    def generate(self, prompt):
        """Returns generated image for given prompt"""
        pass

    def instantiate_pipeline(self, model_id):
        """Returns instantiated pipeline"""
        pipeline = DiffusionPipeline.from_pretrained(
            model_id).to(self.device)
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


if __name__ == "__main__":
    stable_diffusion = StableDiffusion()


