import os
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.io import loadmat

from dataset.base import HugFewShotDataset
from dataset.template import IMAGENET_TEMPLATES_TINY

SUPER_CLASS_NAME = "dog"
DEFAULT_IMAGE_DIR = "/data/zhicai/datasets/fgvc_datasets/stanford_dogs/"

CLASS_NAME = [
    "Chihuahua",
    "Japanese spaniel",
    "Maltese dog",
    "Pekinese",
    "Shih-Tzu",
    "Blenheim spaniel",
    "papillon",
    "toy terrier",
    "Rhodesian ridgeback",
    "Afghan hound",
    "basset",
    "beagle",
    "bloodhound",
    "bluetick",
    "black-and-tan coonhound",
    "Walker hound",
    "English foxhound",
    "redbone",
    "borzoi",
    "Irish wolfhound",
    "Italian greyhound",
    "whippet",
    "Ibizan hound",
    "Norwegian elkhound",
    "otterhound",
    "Saluki",
    "Scottish deerhound",
    "Weimaraner",
    "Staffordshire bullterrier",
    "American Staffordshire terrier",
    "Bedlington terrier",
    "Border terrier",
    "Kerry blue terrier",
    "Irish terrier",
    "Norfolk terrier",
    "Norwich terrier",
    "Yorkshire terrier",
    "wire-haired fox terrier",
    "Lakeland terrier",
    "Sealyham terrier",
    "Airedale",
    "cairn",
    "Australian terrier",
    "Dandie Dinmont",
    "Boston bull",
    "miniature schnauzer",
    "giant schnauzer",
    "standard schnauzer",
    "Scotch terrier",
    "Tibetan terrier",
    "silky terrier",
    "soft-coated wheaten terrier",
    "West Highland white terrier",
    "Lhasa",
    "flat-coated retriever",
    "curly-coated retriever",
    "golden retriever",
    "Labrador retriever",
    "Chesapeake Bay retriever",
    "German short-haired pointer",
    "vizsla",
    "English setter",
    "Irish setter",
    "Gordon setter",
    "Brittany spaniel",
    "clumber",
    "English springer",
    "Welsh springer spaniel",
    "cocker spaniel",
    "Sussex spaniel",
    "Irish water spaniel",
    "kuvasz",
    "schipperke",
    "groenendael",
    "malinois",
    "briard",
    "kelpie",
    "komondor",
    "Old English sheepdog",
    "Shetland sheepdog",
    "collie",
    "Border collie",
    "Bouvier des Flandres",
    "Rottweiler",
    "German shepherd",
    "Doberman",
    "miniature pinscher",
    "Greater Swiss Mountain dog",
    "Bernese mountain dog",
    "Appenzeller",
    "EntleBucher",
    "boxer",
    "bull mastiff",
    "Tibetan mastiff",
    "French bulldog",
    "Great Dane",
    "Saint Bernard",
    "Eskimo dog",
    "malamute",
    "Siberian husky",
    "affenpinscher",
    "basenji",
    "pug",
    "Leonberg",
    "Newfoundland",
    "Great Pyrenees",
    "Samoyed",
    "Pomeranian",
    "chow",
    "keeshond",
    "Brabancon griffon",
    "Pembroke",
    "Cardigan",
    "toy poodle",
    "miniature poodle",
    "standard poodle",
    "Mexican hairless",
    "dingo",
    "dhole",
    "African hunting dog",
]


class StanfordDogDataset(HugFewShotDataset):

    class_names = CLASS_NAME
    super_class_name = SUPER_CLASS_NAME
    num_classes: int = len(class_names)

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

        if split == "train":
            data_mat = loadmat(os.path.join(image_dir, "train_list.mat"))
        else:
            data_mat = loadmat(os.path.join(image_dir, "test_list.mat"))

        image_files = [
            os.path.join(image_dir, "Images", i[0][0]) for i in data_mat["file_list"]
        ]
        imagelabels = data_mat["labels"].squeeze()
        class_to_images = defaultdict(list)

        for image_idx, image_path in enumerate(image_files):
            class_name = self.class_names[imagelabels[image_idx] - 1]
            class_to_images[class_name].append(image_path)

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


class StanfordDogDatasetForT2I(torch.utils.data.Dataset):

    class_names = CLASS_NAME
    super_class_name = SUPER_CLASS_NAME

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
        if split == "train":
            data_mat = loadmat(os.path.join(image_dir, "train_list.mat"))
        else:
            data_mat = loadmat(os.path.join(image_dir, "test_list.mat"))

        image_files = [
            os.path.join(image_dir, "Images", i[0][0]) for i in data_mat["file_list"]
        ]
        imagelabels = data_mat["labels"].squeeze()
        class_to_images = defaultdict(list)
        random.seed(seed)
        np.random.seed(seed)
        if max_train_samples is not None and max_train_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))
        class_to_images = defaultdict(list)

        for image_idx, image_path in enumerate(image_files):
            class_name = self.class_names[imagelabels[image_idx] - 1]
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)
        class_to_ids = {
            key: rng.permutation(len(class_to_images[key])) for key in self.class_names
        }

        # Split 0.9/0.1 as Train/Test subset
        class_to_ids = {
            key: np.array_split(class_to_ids[key], [int(0.5 * len(class_to_ids[key]))])[
                0 if split == "train" else 1
            ]
            for key in self.class_names
        }
        if examples_per_class is not None and examples_per_class > 0:
            class_to_ids = {
                key: ids[:examples_per_class] for key, ids in class_to_ids.items()
            }
        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids]
            for key, ids in class_to_ids.items()
        }
        self.all_images = sum(
            [self.class_to_images[key] for key in self.class_names], []
        )
        self.class2label = {key: i for i, key in enumerate(self.class_names)}
        self.label2class = {v: k for k, v in self.class2label.items()}
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

    ds2 = StanfordDogDataset()
    # ds = DogDatasetForT2I(return_onehot=True)
    print(ds2.class_to_images == ds2.class_to_images)
    print("l")
