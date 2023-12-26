import os
import sys
import random
os.environ["DISABLE_TELEMETRY"] = 'YES'
sys.path.append('../')

from utils import visualize_images
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from semantic_aug.augmentations.textual_inversion import load_embeddings
from utils import DATASET_NAME_MAPPING, AUGMENT, parse_finetuned_ckpt

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
    device = 'cuda:1'
    dataset = 'pascal'
    aug = 'dreambooth-lora-mixup' #'dreambooth-lora-augmentation/mixup" "real-mixup"
    finetune_model_key = 'db_latest_5shot'
    guidance_scale = 7
    strength_list = [0.1,0.3,0.5,0.7,0.9,1.0]


    seed = 0
    random.seed(seed)
    source_label=5
    target_label=10
    for target_label in [4,7,6]:
        for dataset in ['pascal']:
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
                    source_image = train_dataset.get_image_by_idx(train_dataset.label_to_indices[source_label][3])
                    image_list.append(synthesize_images(model,strength,train_dataset,source_label=source_label,target_label=target_label,source_image=source_image)[0])

                outpath=f'figures/cases/{dataset}/{aug}_{source_label}_{target_label}.png'
                visualize_images(image_list,nrow=6,show=False,save=True,outpath=outpath)
