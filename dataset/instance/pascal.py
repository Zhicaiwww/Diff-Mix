import os
import random
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image

from dataset.base import HugFewShotDataset
from dataset.template import IMAGENET_TEMPLATES_TINY

PASCAL_DIR = "/data/zhicai/datasets/VOCdevkit/VOC2012/"

TRAIN_IMAGE_SET = os.path.join(PASCAL_DIR, "ImageSets/Segmentation/train.txt")
VAL_IMAGE_SET = os.path.join(PASCAL_DIR, "ImageSets/Segmentation/val.txt")
DEFAULT_IMAGE_DIR = os.path.join(PASCAL_DIR, "JPEGImages")
DEFAULT_LABEL_DIR = os.path.join(PASCAL_DIR, "SegmentationClass")
DEFAULT_INSTANCE_DIR = os.path.join(PASCAL_DIR, "SegmentationObject")

SUPER_CLASS_NAME = ""
CLASS_NAME = [
    "airplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorcycle",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "television",
]


class PascalDataset(HugFewShotDataset):

    class_names = CLASS_NAME
    num_classes: int = len(class_names)
    super_class_name = SUPER_CLASS_NAME

    def __init__(
        self,
        *args,
        split: str = "train",
        seed: int = 0,
        image_dir: str = DEFAULT_IMAGE_DIR,
        examples_per_class: int = None,
        synthetic_probability: float = 0.5,
        return_onehot: bool = False,
        soft_scaler: float = 0.9,
        synthetic_dir: str = None,
        image_size: int = 512,
        crop_size: int = 448,
        **kwargs,
    ):

        super().__init__(
            *args,
            split=split,
            examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot,
            soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,
            image_size=image_size,
            crop_size=crop_size,
            **kwargs,
        )

        image_set = {"train": TRAIN_IMAGE_SET, "val": VAL_IMAGE_SET}[split]

        with open(image_set, "r") as f:
            image_set_lines = [x.strip() for x in f.readlines()]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        for image_id in image_set_lines:

            labels = os.path.join(DEFAULT_LABEL_DIR, image_id + ".png")
            instances = os.path.join(DEFAULT_INSTANCE_DIR, image_id + ".png")

            labels = np.asarray(Image.open(labels))
            instances = np.asarray(Image.open(instances))

            instance_ids, pixel_loc, counts = np.unique(
                instances, return_index=True, return_counts=True
            )

            counts[0] = counts[-1] = 0  # remove background

            argmax_index = counts.argmax()

            mask = np.equal(instances, instance_ids[argmax_index])
            class_name = self.class_names[labels.flat[pixel_loc[argmax_index]] - 1]

            class_to_images[class_name].append(
                os.path.join(image_dir, image_id + ".jpg")
            )
            class_to_annotations[class_name].append(dict(mask=mask))

        rng = np.random.default_rng(seed)
        class_to_ids = {
            key: rng.permutation(len(class_to_images[key])) for key in self.class_names
        }

        if examples_per_class is not None and examples_per_class > 0:
            class_to_ids = {
                key: ids[:examples_per_class] for key, ids in class_to_ids.items()
            }

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids]
            for key, ids in class_to_ids.items()
        }

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids]
            for key, ids in class_to_ids.items()
        }

        self.class2label = {key: i for i, key in enumerate(self.class_names)}
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.all_images = sum(
            [self.class_to_images[key] for key in self.class_names], []
        )
        self.all_labels = [
            i
            for i, key in enumerate(self.class_names)
            for _ in self.class_to_images[key]
        ]

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.all_labels):
            self.label_to_indices[label].append(i)

    def __len__(self):

        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert("RGB")

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]

    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(
            name=self.class_names[self.all_labels[idx]],
            super_class=self.super_class_name,
        )


class PascalDatasetForT2I(torch.utils.data.Dataset):
    super_class_name = SUPER_CLASS_NAME
    class_names = CLASS_NAME

    def __init__(
        self,
        *args,
        split: str = "train",
        seed: int = 0,
        image_dir: str = DEFAULT_IMAGE_DIR,
        max_train_samples: int = -1,
        class_prompts_ratio: float = 0.5,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
        use_placeholder: bool = False,
        examples_per_class: int = -1,
        **kwargs,
    ):

        super().__init__()
        image_set = {"train": TRAIN_IMAGE_SET, "val": VAL_IMAGE_SET}[split]

        with open(image_set, "r") as f:
            image_set_lines = [x.strip() for x in f.readlines()]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        for image_id in image_set_lines:

            labels = os.path.join(DEFAULT_LABEL_DIR, image_id + ".png")
            instances = os.path.join(DEFAULT_INSTANCE_DIR, image_id + ".png")

            labels = np.asarray(Image.open(labels))
            instances = np.asarray(Image.open(instances))

            instance_ids, pixel_loc, counts = np.unique(
                instances, return_index=True, return_counts=True
            )

            counts[0] = counts[-1] = 0  # remove background

            argmax_index = counts.argmax()

            mask = np.equal(instances, instance_ids[argmax_index])
            class_name = self.class_names[labels.flat[pixel_loc[argmax_index]] - 1]

            class_to_images[class_name].append(
                os.path.join(image_dir, image_id + ".jpg")
            )
            class_to_annotations[class_name].append(dict(mask=mask))

        rng = np.random.default_rng(seed)
        class_to_ids = {
            key: rng.permutation(len(class_to_images[key])) for key in self.class_names
        }

        if examples_per_class is not None and examples_per_class > 0:
            class_to_ids = {
                key: ids[:examples_per_class] for key, ids in class_to_ids.items()
            }

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids]
            for key, ids in class_to_ids.items()
        }

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids]
            for key, ids in class_to_ids.items()
        }

        self.class2label = {key: i for i, key in enumerate(self.class_names)}
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.all_images = sum(
            [self.class_to_images[key] for key in self.class_names], []
        )
        self.all_labels = [
            i
            for i, key in enumerate(self.class_names)
            for _ in self.class_to_images[key]
        ]

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.all_labels):
            self.label_to_indices[label].append(i)
        self.num_classes = len(self.class_names)
        self.class_prompts_ratio = class_prompts_ratio
        self.use_placeholder = use_placeholder
        self.name2placeholder = {}
        self.placeholder2name = {}

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                (
                    transforms.CenterCrop(resolution)
                    if center_crop
                    else transforms.RandomCrop(resolution)
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if random_flip
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):

        return len(self.all_images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        image = self.get_image_by_idx(idx)
        prompt = self.get_prompt_by_idx(idx)
        # label = self.get_label_by_idx(idx)

        return dict(pixel_values=self.transform(image), caption=prompt)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert("RGB")

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]

    def get_prompt_by_idx(self, idx: int) -> int:
        # randomly choose from class name or description

        if self.use_placeholder:
            content = (
                self.name2placeholder[self.label2class[self.get_label_by_idx(idx)]]
                + f"{self.super_class_name}"
            )
        else:
            content = self.label2class[self.get_label_by_idx(idx)]
        prompt = random.choice(IMAGENET_TEMPLATES_TINY).format(content)

        return prompt

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.class_names[self.all_labels[idx]])


if __name__ == "__main__":
    ds = PascalDataset(return_onehot=True)
    ds2 = PascalDatasetForT2I()
    print(ds.class_to_images == ds2.class_to_images)
    print("l")
