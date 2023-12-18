import os
import torch
import huggingface_hub
import sys
sys.path.append('../')
from base.utils import visualize_images
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from semantic_aug.augmentations.textual_inversion import load_embeddings
import numpy as np
import random
import time
from utils import DATASET_NAME_MAPPING, AUGMENT, parse_finetuned_ckpt
os.environ["DISABLE_TELEMETRY"] = 'YES'



DA_LIST = np.array([0.25,0.5,0.75,1.0])
MIX_LIST = np.array([0.5,0.7,0.9])

def synthesize_images(model, strength, train_dataset,source_label=1, target_label=2, source_image=None):
    num = 1
    random.seed(seed)
    target_indice = random.sample(train_dataset.label_to_indices[target_label], 1)[0]

    if source_image is None:
        source_indice = random.sample(train_dataset.label_to_indices[source_label], 1)[0]
        source_image = train_dataset.get_image_by_idx(source_indice)
    target_metadata = train_dataset.get_metadata_by_idx(target_indice)
    image_list = []
    image, _ = model(image=[source_image], label=target_label, strength=strength, metadata=target_metadata)
    return image

if __name__ == '__main__':
    device = 'cuda:3'
    dataset = 'pascal'
    aug = 'dreambooth-lora-mixup' #'dreambooth-lora-augmentation/mixup" "real-mixup"
    finetune_model_key = 'db_latest_5shot'
    guidance_scale = 7
    strength_list = [0.1,0.3,0.5,0.7,0.9,1.0]


    seed = 0
    random.seed(seed)
    source_label=5
    target_label=10
    for aug in ['dreambooth-lora-mixup']:
        train_dataset = DATASET_NAME_MAPPING[dataset](split="train",examples_per_class=5)
        lora_path, embed_path = parse_finetuned_ckpt(dataset=dataset, finetune_model_key=finetune_model_key)

        AUGMENT[aug].pipe=None
        model = AUGMENT[aug](
            embed_path=embed_path,
            lora_path=lora_path, 
            prompt="a photo of a {name}", 
            guidance_scale=guidance_scale,
            mask=False, 
            inverted=False,
            device=device
            )

        image_list = []
        for strength in strength_list:
            start_time = time.time()
            for _ in range(100):
                strength = np.random.choice(MIX_LIST)
                print(strength)
                source_image = train_dataset.get_image_by_idx(train_dataset.label_to_indices[source_label][3])
                image_list.append(synthesize_images(model, strength, train_dataset, source_label=source_label, target_label=target_label, source_image=source_image)[0])
            end_time = time.time()
            images_generated_per_hour = 100 / ((end_time - start_time) / 3600)
            print(f"Images generated for diffmix per hour: {images_generated_per_hour}")
        outpath=f'figures/cases/{dataset}/{aug}_{source_label}_{target_label}.png'
        # visualize_images(image_list,nrow=6,show=False,save=True,outpath=outpath)
