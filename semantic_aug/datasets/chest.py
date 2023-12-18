import sys 
import os
sys.path.append(os.getcwd())
os.environ["http_proxy"]="http://localhost:8890"
os.environ["https_proxy"]="http://localhost:8890"
from semantic_aug.few_shot_dataset import FewShotDataset, HugFewShotDataset
from semantic_aug.datasets.utils import IMAGENET_TEMPLATES_SMALL 
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch
import glob
import random

from PIL import Image
from collections import defaultdict
from datasets import load_dataset

SUPER_CLASS_NAME='chest xray'
DEFAULT_IMAGE_DIR = r"keremberke/chest-xray-classification"


class ChestHugDataset(HugFewShotDataset):

    super_class_name = SUPER_CLASS_NAME 

    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 examples_per_class: int = -1, 
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

        dataset =  load_dataset(image_dir,name='full')
        dataset = dataset.rename_column('labels','label')
        dataset = dataset['train'] if split == 'train' else dataset['test']

        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]

        random.seed(seed)
        np.random.seed(seed)
        if examples_per_class is not None and examples_per_class  > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population or is negative, use random.choices instead")
                    sampled_indices = random.choices(items, k=examples_per_class)
                    
                label_to_indices[key] = sampled_indices 
                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)

        self.dataset = dataset
        self.class2label = self.dataset.features['label']._str2int
        self.label2class = {v: k for k, v in self.class2label.items()} 
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.num_classes = len(self.class_names)
        
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)


    def __len__(self):
        
        return len(self.dataset)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]['label']
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.label2class[self.get_label_by_idx(idx)],super_class = self.super_class_name)


class ChestHugDatasetForT2I(torch.utils.data.Dataset):

    super_class_name = SUPER_CLASS_NAME    

    def __init__(self, *args, 
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
                 **kwargs):

        super().__init__()    

        dataset =  load_dataset(image_dir,name='full')
        dataset = dataset.rename_column('labels','label')
        dataset = dataset['train'] if split == 'train' else dataset['test']

        random.seed(seed)
        np.random.seed(seed)
        if  max_train_samples is not None and max_train_samples > 0:
            dataset = dataset.shuffle(seed=seed).select(range(max_train_samples))
        if examples_per_class is not None and examples_per_class > 0:
            all_labels = dataset['label']
            label_to_indices = defaultdict(list)
            for i, label in enumerate(all_labels):
                label_to_indices[label].append(i)

            _all_indices = []
            for key, items in label_to_indices.items():
                try:
                    sampled_indices = random.sample(items, examples_per_class)
                except ValueError:
                    print(f"{key}: Sample larger than population or is negative, use random.choices instead")
                    sampled_indices = random.choices(items, k=examples_per_class)
                    
                label_to_indices[key] = sampled_indices 
                _all_indices.extend(sampled_indices)
            dataset = dataset.select(_all_indices)

        self.dataset = dataset
        self.class2label = self.dataset.features['label']._str2int
        self.label2class = {v: k for k, v in self.class2label.items()} 
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.num_classes = len(self.class_names)
        self.use_placeholder = use_placeholder        
        self.name2placeholder = None
        self.placeholder2name = None
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)

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
        
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        image = self.get_image_by_idx(idx)
        prompt = self.get_prompt_by_idx(idx)
        # label = self.get_label_by_idx(idx)

        return dict(pixel_values=self.transform(image), caption = prompt)
    
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]['label']
    
    def get_prompt_by_idx(self, idx: int) -> int:
        # randomly choose from class name or description
        if self.use_placeholder:
            content = self.name2placeholder[self.label2class[self.dataset[idx]['label']]] + f' {self.super_class_name}'
        else:
            content = self.label2class[self.dataset[idx]['label']] 
        prompt =  random.choice(IMAGENET_TEMPLATES_SMALL).format(content)
        
        return prompt

    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.label2class[self.get_label_by_idx(idx)])

if __name__ == '__main__':
    ds_train = ChestHugDatasetForT2I(max_train_samples=10)
    print(ds_train[0])