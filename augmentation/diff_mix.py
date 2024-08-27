import os
from typing import Callable, Tuple

import torch
from PIL import Image
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer

from augmentation.base_augmentation import GenerativeMixup
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from diffusers.utils import logging

os.environ["WANDB_DISABLED"] = "true"
ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


def load_diffmix_embeddings(
    embed_path: str,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device="cuda",
):

    embedding_ckpt = torch.load(embed_path, map_location="cpu")
    learned_embeds_dict = embedding_ckpt["learned_embeds_dict"]
    name2placeholder = embedding_ckpt["name2placeholder"]
    placeholder2name = embedding_ckpt["placeholder2name"]

    name2placeholder = {
        k.replace("/", " ").replace("_", " "): v for k, v in name2placeholder.items()
    }
    placeholder2name = {
        v: k.replace("/", " ").replace("_", " ") for k, v in name2placeholder.items()
    }

    for token, token_embedding in learned_embeds_dict.items():

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

    return name2placeholder, placeholder2name


def identity(*args):
    return args


class IdentityMap:
    def __getitem__(self, key):
        return key


class DreamboothLoraMixup(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        lora_path: str,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        embed_path: str = None,
        prompt: str = "a photo of a {name}",
        format_name: Callable = format_name,
        guidance_scale: float = 7.5,
        disable_safety_checker: bool = True,
        revision: str = None,
        device="cuda",
        **kwargs,
    ):

        super(DreamboothLoraMixup, self).__init__()

        if DreamboothLoraMixup.pipe is None:

            PipelineClass = StableDiffusionImg2ImgPipeline

            DreamboothLoraMixup.pipe = PipelineClass.from_pretrained(
                model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to(device)

            scheduler = DPMSolverMultistepScheduler.from_config(
                DreamboothLoraMixup.pipe.scheduler.config, local_files_only=True
            )
            self.placeholder2name = {}
            self.name2placeholder = {}
            if embed_path is not None:
                self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(
                    embed_path,
                    DreamboothLoraMixup.pipe.text_encoder,
                    DreamboothLoraMixup.pipe.tokenizer,
                )
            if lora_path is not None:
                DreamboothLoraMixup.pipe.load_lora_weights(lora_path)
            DreamboothLoraMixup.pipe.scheduler = scheduler

            print(f"successfuly load lora weights from {lora_path}! ! ! ")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name

    def forward(
        self,
        image: Image.Image,
        label: int,
        metadata: dict,
        strength: float = 0.5,
        resolution=512,
    ) -> Tuple[Image.Image, int]:

        canvas = [img.resize((resolution, resolution), Image.BILINEAR) for img in image]
        name = metadata.get("name", "")

        if self.name2placeholder is not None:
            name = self.name2placeholder[name]
        if metadata.get("super_class", None) is not None:
            name = name + " " + metadata.get("super_class", "")
        prompt = self.prompt.format(name=name)

        print(prompt)

        kwargs = dict(
            image=canvas,
            prompt=[prompt],
            strength=strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=25,
            num_images_per_prompt=len(canvas),
        )

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


class DreamboothLoraGeneration(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        lora_path: str,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        embed_path: str = None,
        prompt: str = "a photo of a {name}",
        format_name: Callable = format_name,
        guidance_scale: float = 7.5,
        disable_safety_checker: bool = True,
        revision: str = None,
        device="cuda",
        **kwargs,
    ):

        super(DreamboothLoraGeneration, self).__init__()

        if DreamboothLoraGeneration.pipe is None:

            PipelineClass = StableDiffusionPipeline

            DreamboothLoraGeneration.pipe = PipelineClass.from_pretrained(
                model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=True,
                torch_dtype=torch.float16,
            ).to(device)

            scheduler = DPMSolverMultistepScheduler.from_config(
                DreamboothLoraGeneration.pipe.scheduler.config, local_files_only=True
            )
            self.placeholder2name = None
            self.name2placeholder = None
            if embed_path is not None:
                self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(
                    embed_path,
                    DreamboothLoraGeneration.pipe.text_encoder,
                    DreamboothLoraGeneration.pipe.tokenizer,
                )
            if lora_path is not None:
                DreamboothLoraGeneration.pipe.load_lora_weights(lora_path)
            DreamboothLoraGeneration.pipe.scheduler = scheduler

            print(f"successfuly load lora weights from {lora_path}! ! ! ")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name

    def forward(
        self,
        image: Image.Image,
        label: int,
        metadata: dict,
        strength: float = 0.5,
        resolution=512,
    ) -> Tuple[Image.Image, int]:

        name = metadata.get("name", "")

        if self.name2placeholder is not None:
            name = self.name2placeholder[name]
        if metadata.get("super_class", None) is not None:
            name = name + " " + metadata.get("super_class", "")
        prompt = self.prompt.format(name=name)

        print(prompt)

        kwargs = dict(
            prompt=[prompt],
            guidance_scale=self.guidance_scale,
            num_inference_steps=25,
            num_images_per_prompt=len(image),
            height=resolution,
            width=resolution,
        )

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None
                and outputs.nsfw_content_detected[0]
            )

        canvas = []
        for out in outputs.images:
            canvas.append(out)
        return canvas, label
