import os
import random
import sys

os.environ["DISABLE_TELEMETRY"] = "YES"
sys.path.append("../")

from augmentation import AUGMENT_METHODS
from dataset import DATASET_NAME_MAPPING
from utils.misc import finetuned_ckpt_dir
from utils.visualization import visualize_images


def synthesize_images(
    model, strength, train_dataset, source_label=1, target_label=2, source_image=None
):
    num = 1
    random.seed(seed)
    target_indice = random.sample(train_dataset.label_to_indices[target_label], 1)[0]

    if source_image is None:
        source_indice = random.sample(train_dataset.label_to_indices[source_label], 1)[
            0
        ]
        source_image = train_dataset.get_image_by_idx(source_indice)
    target_metadata = train_dataset.get_metadata_by_idx(target_indice)
    image_list = []
    image, _ = model(
        image=[source_image],
        label=target_label,
        strength=strength,
        metadata=target_metadata,
    )
    return image


if __name__ == "__main__":
    device = "cuda:1"
    dataset = "pascal"
    aug = "diff-mix"  #'diff-aug/mixup" "real-mix"
    finetuned_ckpt = "db_latest_5shot"
    guidance_scale = 7
    strength_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    seed = 0
    random.seed(seed)
    source_label = 5
    for target_label in [4, 7, 6]:
        for dataset in ["pascal"]:
            for aug in ["diff-mix"]:
                train_dataset = DATASET_NAME_MAPPING[dataset](
                    split="train", examples_per_class=5
                )
                lora_path, embed_path = finetuned_ckpt_dir(
                    dataset=dataset, finetuned_ckpt=finetuned_ckpt
                )

                AUGMENT_METHODS[aug].pipe = None
                model = AUGMENT_METHODS[aug](
                    embed_path=embed_path,
                    lora_path=lora_path,
                    prompt="a photo of a {name}",
                    guidance_scale=guidance_scale,
                    mask=False,
                    inverted=False,
                    device=device,
                )

                image_list = []
                for strength in strength_list:
                    source_image = train_dataset.get_image_by_idx(
                        train_dataset.label_to_indices[source_label][3]
                    )
                    image_list.append(
                        synthesize_images(
                            model,
                            strength,
                            train_dataset,
                            source_label=source_label,
                            target_label=target_label,
                            source_image=source_image,
                        )[0]
                    )

                outpath = (
                    f"figures/cases/{dataset}/{aug}_{source_label}_{target_label}.png"
                )
                visualize_images(
                    image_list, nrow=6, show=False, save=True, outpath=outpath
                )
