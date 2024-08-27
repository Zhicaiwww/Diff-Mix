import os
import sys
from typing import Any, Callable, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import maximum_filter
from torch import autocast

from augmentation.base_augmentation import GenerativeAugmentation, GenerativeMixup
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from diffusers.utils import logging


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


class RealGeneration(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
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
        **kwargs,
    ):

        super(RealGeneration, self).__init__()

        if RealGeneration.pipe is None:

            PipelineClass = StableDiffusionPipeline

            RealGeneration.pipe = PipelineClass.from_pretrained(
                model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to(device)
            scheduler = DPMSolverMultistepScheduler.from_config(
                RealGeneration.pipe.scheduler.config
            )
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

    def forward(
        self,
        image: Image.Image,
        label: int,
        metadata: dict,
        strength: float = 0.5,
        resolution=512,
    ) -> Tuple[Image.Image, int]:

        name = self.format_name(metadata.get("name", ""))
        prompt = self.prompt.format(name=name)

        if self.mask:
            assert "mask" in metadata, "mask=True but no mask present in metadata"

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
                maximum_filter(np.array(mask_image), size=self.mask_grow_radius)
            )

            if self.inverted:

                mask_image = ImageOps.invert(mask_image.convert("L")).convert("1")

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


class RealGuidance(GenerativeAugmentation):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        prompt: str = "a photo of a {name}",
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        mask: bool = False,
        inverted: bool = False,
        mask_grow_radius: int = 16,
        erasure_ckpt_path: str = None,
        disable_safety_checker: bool = True,
        **kwargs,
    ):

        super(RealGuidance, self).__init__()

        if RealGuidance.pipe is None:

            PipelineClass = (
                StableDiffusionInpaintPipeline
                if mask
                else StableDiffusionImg2ImgPipeline
            )

            self.pipe = PipelineClass.from_pretrained(
                model_path,
                use_auth_token=True,
                revision="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale

        self.mask = mask
        self.inverted = inverted
        self.mask_grow_radius = mask_grow_radius

        self.erasure_ckpt_path = erasure_ckpt_path
        self.erasure_word_name = None

    def forward(
        self, image: Image.Image, label: int, metadata: dict
    ) -> Tuple[Image.Image, int]:

        canvas = image.resize((512, 512), Image.BILINEAR)
        prompt = self.prompt.format(name=metadata.get("name", ""))

        if self.mask:
            assert "mask" in metadata, "mask=True but no mask present in metadata"

        word_name = metadata.get("name", "").replace(" ", "")

        if self.erasure_ckpt_path is not None and (
            self.erasure_word_name is None or self.erasure_word_name != word_name
        ):

            self.erasure_word_name = word_name
            ckpt_name = "method_full-sg_3-ng_1-iter_1000-lr_1e-05"

            ckpt_path = os.path.join(
                self.erasure_ckpt_path,
                f"compvis-word_{word_name}-{ckpt_name}",
                f"diffusers-word_{word_name}-{ckpt_name}.pt",
            )

            self.pipe.unet.load_state_dict(torch.load(ckpt_path, map_location="cuda"))

        kwargs = dict(
            image=canvas,
            prompt=[prompt],
            strength=self.strength,
            guidance_scale=self.guidance_scale,
        )

        if self.mask:  # use focal object mask

            mask_image = Image.fromarray(
                (np.where(metadata["mask"], 255, 0)).astype(np.uint8)
            ).resize((512, 512), Image.NEAREST)

            mask_image = Image.fromarray(
                maximum_filter(np.array(mask_image), size=self.mask_grow_radius)
            )

            if self.inverted:

                mask_image = ImageOps.invert(mask_image.convert("L")).convert("1")

            kwargs["mask_image"] = mask_image

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None
                and outputs.nsfw_content_detected[0]
            )

        canvas = outputs.images[0].resize(image.size, Image.BILINEAR)

        return canvas, label
