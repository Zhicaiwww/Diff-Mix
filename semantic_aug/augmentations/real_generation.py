import sys
sys.path.append('/data/zhicai/code/da-fusion/')
from semantic_aug.generative_augmentation import GenerativeMixup
from diffusers import StableDiffusionImg2ImgPipeline,DPMSolverMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
from transformers import (
    CLIPFeatureExtractor, 
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers.utils import logging
from PIL import Image, ImageOps

from typing import Any, Tuple, Callable
from torch import autocast
from scipy.ndimage import maximum_filter

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["http_proxy"]="http://localhost:8890"
os.environ["https_proxy"]="http://localhost:8890"
os.environ["WANDB_DISABLED"] = "true"

def format_name(name):
    return f"<{name.replace(' ', '_')}>"

class RealGeneration(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(self, 
                 model_path: str = "runwayml/stable-diffusion-v1-5",
                 prompt: str = "a photo of a {name}",
                 format_name: Callable = format_name,
                 guidance_scale: float = 7.5,
                 mask: bool = False,
                 inverted: bool = False,
                 mask_grow_radius: int = 16,
                 disable_safety_checker: bool = True,
                 revision: str = None,
                 device="cuda", 
                 **kwargs):

        super(RealGeneration, self).__init__()

        if RealGeneration.pipe is None:

            PipelineClass = StableDiffusionPipeline


            RealGeneration.pipe = PipelineClass.from_pretrained(
                model_path, use_auth_token=True,
                revision=revision, 
                local_files_only=True,
                torch_dtype=torch.float16
            ).to(device)
            scheduler = DPMSolverMultistepScheduler.from_config(RealGeneration.pipe.scheduler.config)
            RealGeneration.pipe.scheduler = scheduler
            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius


    def forward(self, image: Image.Image, label: int, 
                metadata: dict, strength: float=0.5, resolution=512) -> Tuple[Image.Image, int]:

        name = self.format_name(metadata.get("name", ""))
        prompt = self.prompt.format(name=name)

        if self.mask: assert "mask" in metadata, \
            "mask=True but no mask present in metadata"
        
        # word_name = metadata.get("name", "").replace(" ", "")

        kwargs = dict(
            prompt=[prompt], 
            guidance_scale=self.guidance_scale,
            num_inference_steps=25,
            num_images_per_prompt=len(image),
            height=resolution,
            width=resolution,
        )

        if self.mask:  # use focal object mask
            # TODO
            mask_image = metadata["mask"].resize((512, 512), Image.NEAREST)
            mask_image = Image.fromarray(
                maximum_filter(np.array(mask_image), 
                               size=self.mask_grow_radius))

            if self.inverted:

                mask_image = ImageOps.invert(
                    mask_image.convert('L')).convert('1')

            kwargs["mask_image"] = mask_image

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None 
                and outputs.nsfw_content_detected[0]
            )

        canvas = []
        for orig, out in zip(image, outputs.images):
            canvas.append(out.resize(orig.size, Image.BILINEAR))

        return canvas, label