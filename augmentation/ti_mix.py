from typing import Any, Callable, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import maximum_filter
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer

from augmentation.base_augmentation import GenerativeMixup
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from diffusers.utils import logging

ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."


def load_embeddings(
    embed_path: str,
    model_path: str = "runwayml/stable-diffusion-v1-5",
    revision: str = "39593d5650112b4cc580433f6b0435385882d819",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, use_auth_token=True, revision=revision, subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        model_path, use_auth_token=True, revision=revision, subfolder="text_encoder"
    )

    for token, token_embedding in torch.load(embed_path, map_location="cpu").items():

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = token_embedding.to(
            embeddings.weight.dtype
        )

    return tokenizer, text_encoder


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


class TextualInversionMixup(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        embed_path: str,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        prompt: str = "a photo of a {name}",
        format_name: Callable = format_name,
        guidance_scale: float = 7.5,
        mask: bool = False,
        inverted: bool = False,
        mask_grow_radius: int = 16,
        disable_safety_checker: bool = True,
        revision: str = "39593d5650112b4cc580433f6b0435385882d819",
        device="cuda",
        **kwargs,
    ):

        super().__init__()

        if TextualInversionMixup.pipe is None:

            PipelineClass = (
                StableDiffusionInpaintPipeline
                if mask
                else StableDiffusionImg2ImgPipeline
            )

            tokenizer, text_encoder = load_embeddings(
                embed_path, model_path=model_path, revision=revision
            )

            TextualInversionMixup.pipe = PipelineClass.from_pretrained(
                model_path,
                use_auth_token=True,
                revision=revision,
                torch_dtype=torch.float16,
            ).to(device)
            scheduler = DPMSolverMultistepScheduler.from_config(
                TextualInversionMixup.pipe.scheduler.config
            )
            TextualInversionMixup.pipe.scheduler = scheduler
            self.pipe.tokenizer = tokenizer
            self.pipe.text_encoder = text_encoder.to(device)

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

        self.erasure_word_name = None

    def forward(
        self, image: Image.Image, label: int, metadata: dict, strength: float = 0.5
    ) -> Tuple[Image.Image, int]:

        canvas = image.resize((512, 512), Image.BILINEAR)
        name = self.format_name(metadata.get("name", ""))
        prompt = self.prompt.format(name=name)

        if self.mask:
            assert "mask" in metadata, "mask=True but no mask present in metadata"

        word_name = metadata.get("name", "").replace(" ", "")

        kwargs = dict(
            image=canvas,
            prompt=[prompt],
            strength=strength,
            guidance_scale=self.guidance_scale,
        )

        if self.mask:  # use focal object mask
            # TODO
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
