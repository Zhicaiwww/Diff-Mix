import glob
import os
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_from_disk
from PIL import Image
from scipy.io import loadmat

from dataset.base import HugFewShotDataset
from dataset.template import IMAGENET_TEMPLATES_TINY

SUPER_CLASS_NAME = "flower"
DEFAULT_IMAGE_DIR = "/data/zhicai/datasets/fgvc_datasets/flowers102"
DUFAULT_HUG_REPO = "nelorth/oxford-flowers"
DEFAULT_IMAGE_LOCAL_DIR = r"/home/zhicai/.cache/huggingface/local/oxford-flowers"

CLASS_NAME = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen ",
    "watercress",
    "canna lily",
    "hippeastrum ",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]

class Flowers102Dataset(HugFewShotDataset):

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

        imagelabels = loadmat(os.path.join(image_dir, "imagelabels.mat"))["labels"][0]
        image_files = sorted(list(glob.glob(os.path.join(image_dir, "jpg/*.jpg"))))

        class_to_images = defaultdict(list)

        for image_idx, image_path in enumerate(image_files):
            class_name = self.class_names[imagelabels[image_idx] - 1]
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)
        class_to_ids = {
            key: rng.permutation(len(class_to_images[key])) for key in self.class_names
        }

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


class Flowers102HugDataset(HugFewShotDataset):

    class_names = CLASS_NAME
    super_class_name = SUPER_CLASS_NAME
    num_classes: int = len(class_names)

    def __init__(
        self,
        *args,
        split: str = "train",
        seed: int = 0,
        image_dir: str = DEFAULT_IMAGE_LOCAL_DIR,
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
            dataset = load_from_disk(image_dir)["train"]
        else:
            dataset = load_from_disk(image_dir)["test"]

        random.seed(seed)
        np.random.seed(seed)
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset["label"]
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(
                        f"{key}: Sample {examples_per_class} larger than population {len(items)} or is negative, use random.choices instead"
                    )
                    sampled_indices = random.choices(items, k=examples_per_class)

                label_to_indices[key] = sampled_indices
                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)

        self.dataset = dataset
        self.class2label = self.dataset.features["label"]._str2int
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.class_names = [
            name.replace("/", " ") for name in dataset.features["label"].names
        ]
        self.num_classes = len(self.class_names)

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset["label"]):
            self.label_to_indices[label].append(i)

    def __len__(self):

        return len(self.dataset)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]["image"].convert("RGB")

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]["label"]

    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.label2class[self.get_label_by_idx(idx)])


class FlowersDatasetForT2I(torch.utils.data.Dataset):
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
        imagelabels = loadmat(os.path.join(image_dir, "imagelabels.mat"))["labels"][0]
        image_files = sorted(list(glob.glob(os.path.join(image_dir, "jpg/*.jpg"))))

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


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


class FlowersImbalanceDataset(Flowers102Dataset):
    super_class_name = SUPER_CLASS_NAME

    def __init__(
        self,
        *args,
        split: str = "train",
        seed: int = 0,
        examples_per_class: int = -1,
        synthetic_probability: float = 0.5,
        return_onehot: bool = False,
        soft_scaler: float = 0.9,
        synthetic_dir: str = None,
        image_size: int = 512,
        crop_size: int = 448,
        use_randaug: bool = False,
        imbalance_factor: float = 0.01,
        weighted_alpha: int = 1,
        use_weighted_syn: bool = False,
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

        self.use_randaug = use_randaug
        self.weighted_alpha = weighted_alpha
        if split == "train":
            self.gen_imbalanced_data(imbalance_factor)
            self.normalized_probabilities = (
                self.weighted_prob() if use_weighted_syn else None
            )
            if synthetic_dir is not None:
                self.syn_label_to_indices = self.create_bird_indices_dict()

    def weighted_prob(self):
        cls_num_list = self.get_cls_num_list()
        probabilities = [1 / (num + 1) for num in cls_num_list]
        total_prob = sum(probabilities)
        normalized_probabilities = [prob / total_prob for prob in probabilities]
        print("\nUsing weighted probability ! \n")
        return normalized_probabilities

    def create_bird_indices_dict(self):
        bird_indices = {}
        arr = self.meta_df["Second Directory"].values
        for label, bird_name in self.label2class.items():
            indices = np.where(arr == bird_name)[0]
            bird_indices[label] = indices
        return bird_indices

    def gen_imbalanced_data(self, imb_factor):
        # sort dataset from most to least common class
        img_average = len(self.all_images) / self.num_classes
        org_num = len(self.all_images)
        label_to_indices = defaultdict(list)
        all_indices = []
        for sorted_idx, (sorted_label, indices) in enumerate(
            sorted(self.label_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
        ):
            num_imgs_cur_label = len(indices)
            num = img_average * (imb_factor ** (sorted_idx / (self.num_classes - 2.0)))
            unbalance_indices = random.sample(indices, max(int(num), 1))
            label_to_indices[sorted_label] = unbalance_indices
            all_indices.extend(unbalance_indices)
        self.all_images = [self.all_images[i] for i in all_indices]
        self.all_labels = [self.all_labels[i] for i in all_indices]
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.all_labels):
            self.label_to_indices[label].append(i)
        cur_num = len(self)
        print(
            f"Dataset size filtered from {org_num} to {cur_num} with imbalance factor {imb_factor}"
        )

    def get_cls_num_list(self):
        cls_num_list = [
            len(self.label_to_indices[label]) for label in range(self.num_classes)
        ]
        return cls_num_list

    def get_weighted_sampler(self):

        cls_num_list = self.get_cls_num_list()
        print(cls_num_list)
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.all_labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(
            samples_weight, len(self), replacement=True
        )
        return sampler

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if (
            self.synthetic_dir is not None
            and np.random.uniform() < self.synthetic_probability
        ):

            if self.normalized_probabilities is not None:
                cls = np.random.choice(
                    self.num_classes, p=self.normalized_probabilities
                )
                syn_idx = random.choice(self.syn_label_to_indices[cls])
                image, label = self.get_syn_item(syn_idx)
            else:
                syn_idx = np.random.choice(self.syn_nums)
                image, label = self.get_syn_item(syn_idx)
            if isinstance(image, str):
                image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)
        if self.return_onehot:
            if isinstance(label, (int, np.int64)):
                label = onehot(self.num_classes, label)

        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                image = self.transform[0](image)
            else:
                image = self.transform[1](image)
        else:
            if self.transform is not None:
                image = self.transform(image)
        return dict(pixel_values=image, label=label)


class FlowersImbalanceDatasetForT2I(FlowersDatasetForT2I):
    def __init__(
        self,
        *args,
        split: str = "train",
        seed: int = 0,
        max_train_samples: int = -1,
        class_prompts_ratio: float = 0.5,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
        use_placeholder: bool = False,
        examples_per_class: int = -1,
        imbalance_factor: float = 0.01,
        weighted_alpha: int = 1,
        **kwargs,
    ):
        super().__init__(
            *args,
            split=split,
            seed=seed,
            max_train_samples=max_train_samples,
            class_prompts_ratio=class_prompts_ratio,
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
            use_placeholder=use_placeholder,
            examples_per_class=examples_per_class,
            **kwargs,
        )

        self.weighted_alpha = weighted_alpha
        self.gen_imbalanced_data(imbalance_factor)

    def gen_imbalanced_data(self, imb_factor):
        # sort dataset from most to least common class
        img_average = len(self.all_images) / self.num_classes
        org_num = len(self.all_images)
        label_to_indices = defaultdict(list)
        all_indices = []
        for sorted_idx, (sorted_label, indices) in enumerate(
            sorted(self.label_to_indices.items(), key=lambda x: len(x[1]), reverse=True)
        ):
            num_imgs_cur_label = len(indices)
            num = img_average * (imb_factor ** (sorted_idx / (self.num_classes - 2.0)))
            unbalance_indices = random.sample(indices, max(int(num), 1))
            label_to_indices[sorted_label] = unbalance_indices
            all_indices.extend(unbalance_indices)
        self.all_images = [self.all_images[i] for i in all_indices]
        self.all_labels = [self.all_labels[i] for i in all_indices]
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.all_labels):
            self.label_to_indices[label].append(i)
        cur_num = len(self)
        print(
            f"Dataset size filtered from {org_num} to {cur_num} with imbalance factor {imb_factor}"
        )
