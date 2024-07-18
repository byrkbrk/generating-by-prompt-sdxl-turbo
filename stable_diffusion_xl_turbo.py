import os
import torch
from diffusers import DiffusionPipeline



class StableDiffusion(object):
    def __init__(self, 
                 model_id="stabilityai/sdxl-turbo"
                 ):
        self.module_dir = os.path.dirname(__file__)
        self.device = self.initialize_device()
        self.pipeline = self.instantiate_pipeline(model_id)

    def generate(self, prompt):
        """Returns generated image for given prompt"""
        pass

    def instantiate_pipeline(self, model_id):
        """Returns instantiated pipeline"""
        pipeline = DiffusionPipeline(
            model_id
        )
        return pipeline
