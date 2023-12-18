import sys
sys.path.append('/data/zhicai/code/da-fusion')
from semantic_aug.few_shot_dataset import FewShotDataset, HugFewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import glob
import os
import random

from scipy.io import loadmat
from PIL import Image
from collections import defaultdict
from datasets import load_from_disk
from semantic_aug.datasets.utils import IMAGENET_TEMPLATES_SMALL 

DEFAULT_IMAGE_DIR = "/data/zhicai/datasets/fgvc_datasets/caltech101/101_ObjectCategories"
SUPER_CLASS_NAME = ''
CLASS_NAME = ['accordion', 'airplanes', 'anchor', 'ant', 
    'background google', 'barrel', 'bass', 'beaver', 'binocular', 
    'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 
    'cannon', 'car side', 'ceiling fan', 'cellphone', 'chair', 
    'chandelier', 'cougar body', 'cougar face', 'crab', 'crayfish', 
    'crocodile', 'crocodile head', 'cup', 'dalmatian', 'dollar bill', 
    'dolphin', 'dragonfly', 'electric guitar', 'elephant', 'emu', 
    'euphonium', 'ewer', 'faces', 'faces easy', 'ferry', 'flamingo', 
    'flamingo head', 'garfield', 'gerenuk', 'gramophone', 'grand piano', 
    'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 
    'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 
    'leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 
    'menorah', 'metronome', 'minaret', 'motorbikes', 'nautilus', 
    'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 
    'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 
    'scissors', 'scorpion', 'sea horse', 'snoopy', 'soccer ball', 
    'stapler', 'starfish', 'stegosaurus', 'stop sign', 'strawberry', 
    'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 
    'wheelchair', 'wild cat', 'windsor chair', 'wrench', 'yin yang']

class Caltech101Dataset(HugFewShotDataset):

    class_names = CLASS_NAME
    super_class_name = SUPER_CLASS_NAME
    num_classes: int = len(class_names)

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 examples_per_class: int = None, 
                 synthetic_probability: float = 0.5,
                 return_onehot: bool = False,
                 soft_scaler: float = 0.9,
                 synthetic_dir: str = None,
                 image_size: int = 512,
                 crop_size: int = 448, **kwargs):

        super().__init__(
            *args,split=split, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot, soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,image_size=image_size, crop_size=crop_size, **kwargs)  

        class_to_images = defaultdict(list)

        for image_path in glob.glob(os.path.join(image_dir, "*/*.jpg")):
            class_name = image_path.split("/")[-2].lower().replace("_", " ")
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}

        if examples_per_class is not None and examples_per_class > 0:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}
        self.class2label = {key: i for i, key in enumerate(self.class_names)}
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])
        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.all_labels):
            self.label_to_indices[label].append(i)

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]],super_class=self.super_class_name)



class Caltech101DatasetForT2I(torch.utils.data.Dataset):

    class_names = CLASS_NAME
    super_class_name = SUPER_CLASS_NAME
    
    def __init__(self, *args, split: str = "train",
                 seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 max_train_samples: int = -1,
                 class_prompts_ratio: float = 0.5,
                 resolution: int = 512,
                 center_crop: bool = False,
                 random_flip: bool = False,
                 use_placeholder: bool = False,
                 examples_per_class: int = -1, 
                 **kwargs):

        super().__init__()    
        class_to_images = defaultdict(list)

        for image_path in glob.glob(os.path.join(image_dir, "*/*.jpg")):
            class_name = image_path.split("/")[-2].lower().replace("_", " ")
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}

        if examples_per_class is not None and examples_per_class > 0:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}
        self.class2label = {key: i for i, key in enumerate(self.class_names)}
        self.label2class = {v: k for k, v in self.class2label.items()}
        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])
        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.all_labels):
            self.label_to_indices[label].append(i)

        self.num_classes = len(self.class_names)
        self.class_prompts_ratio = class_prompts_ratio
        self.use_placeholder = use_placeholder        
        self.name2placeholder = None
        self.placeholder2name = None
        
        self.transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
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

        return dict(pixel_values=self.transform(image), caption = prompt)
    
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_prompt_by_idx(self, idx: int) -> int:
        # randomly choose from class name or description

        if self.use_placeholder:
            content = self.name2placeholder[self.label2class[self.get_label_by_idx(idx)]] + f'{self.super_class_name}'
        else:
            content = self.label2class[self.get_label_by_idx(idx)]
        prompt =  random.choice(IMAGENET_TEMPLATES_SMALL).format(content)
        
        return prompt

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.class_names[self.all_labels[idx]])


if __name__ == "__main__" :
    ds = Caltech101DatasetForT2I(return_onehot=True)
    ds2 = Caltech101Dataset()
    print(ds.class_to_images == ds2.class_to_images)
    print('l')
     