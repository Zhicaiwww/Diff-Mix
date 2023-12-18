from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Union, List
from torch.utils.data import Dataset
from collections import defaultdict
from itertools import product
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import torch
import pandas as pd
import numpy as np
import abc
import random
import os
import math

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class SyntheticDataset(Dataset):
    def __init__(self,
                synthetic_dir: Union[str , List[str]] = None,
                synthetic_meta_type: str = 'csv',
                soft_power: int = 1,
                soft_scaler: float = 1,
                num_syn_seeds: int = 999,
                image_size: int = 512,
                crop_size: int = 448,
                class2label: dict =  None,
                csv_file: str = 'meta.csv',
                ) -> None:
        super().__init__()
        self.synthetic_dir = synthetic_dir
        self.num_syn_seeds = num_syn_seeds # number of seeds to generate synthetic data
        self.soft_power = soft_power
        self.soft_scaler = soft_scaler
        self.class_names = None
        if synthetic_meta_type == 'csv':
            self.csv_file = csv_file
            self.parse_syn_data_pd(synthetic_dir)
            self.get_syn_item = self.get_syn_item_pd
        else:
            self.parse_syn_data_pt(synthetic_dir)
            self.get_syn_item = self.get_syn_item_pt

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform = test_transform
        self.class2label = {name:i for i, name in enumerate(self.class_names)} if class2label is None else class2label
        self.num_classes = len(self.class2label.keys())

    def set_transform(self, transform) -> None:
        self.transform = transform

    def parse_syn_data_pt(self, synthetic_dir, filter=True) -> None:
        meta_dir = os.path.join(synthetic_dir, "baseline_top3_logits.pt")
        self.meta_pt = torch.load(meta_dir)
        self.syn_nums = len(self.meta_pt['path'])    

    def parse_syn_data_pd(self, synthetic_dir, filter=True) -> None:
        if isinstance(synthetic_dir, list):
            pass
        elif isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]
        else:
            raise NotImplementedError('Not supported type')
        meta_df_list = []
        for _dir in synthetic_dir:
            meta_dir = os.path.join(_dir, self.csv_file)
            meta_df = pd.read_csv(meta_dir)
            meta_df.loc[:,'Path'] = meta_df['Path'].apply(lambda x: os.path.join(_dir,'data',x))
            meta_df_list.append(meta_df)
        self.meta_df = pd.concat(meta_df_list).reset_index(drop=True)
        self.syn_nums = len(self.meta_df)
        self.class_names = list(set(self.meta_df['First Directory'].values))
        print(f"Syn numbers: {self.syn_nums}\n")
    
    def get_syn_item_pt(self, idx: int):
        path = self.meta_pt['path'][idx]
        indices = self.meta_pt['topk_indices'][idx]
        logits = self.meta_pt['topk_logits'][idx]
        onehot_label = torch.zeros(self.num_classes)
        onehot_label[indices] =  torch.nn.functional.softmax(torch.tensor(logits),dim=0)
        onehot_label = self.soft_scaler * onehot_label
        return os.path.join(self.synthetic_dir,'data', path), onehot_label

    def get_syn_item_pd(self, idx: int):
        df_data = self.meta_df.iloc[idx]
        src_label = self.class2label[df_data['First Directory']]
        tar_label = self.class2label[df_data['Second Directory']]
        path = df_data['Path']
        strength = df_data['Strength']
        onehot_label = torch.zeros(self.num_classes)
        onehot_label[src_label] += self.soft_scaler * (1 - math.pow(strength,self.soft_power))
        onehot_label[tar_label] += self.soft_scaler * math.pow(strength,self.soft_power)
        return path, onehot_label

    def get_syn_item_pd_raw(self, idx: int):
        df_data = self.meta_df.iloc[idx]
        src_label = self.class2label[df_data['First Directory']]
        tar_label = self.class2label[df_data['Second Directory']]
        path = df_data['Path']
        return path, src_label, tar_label
    
    def __len__(self) -> int:
        return self.syn_nums

    def get_image_by_idx(self, idx: int) -> Image.Image:
        image, _ = self.get_syn_item(idx)
        return Image.open(image).convert('RGB')
    
    def get_label_by_idx(self, idx: int) -> int:
        _, label = self.get_syn_item(idx)
        return label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, src_label, target_label = self.get_syn_item_pd_raw(idx)
        if isinstance(image, str): image = Image.open(image).convert('RGB')
        return {'pixel_values':self.transform(image), 'src_label': src_label, 'tar_label': target_label}

class FewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None

    def __init__(self, examples_per_class: int = None, 
                 generative_aug: GenerativeAugmentation = None, 
                 synthetic_probability: float = 0.5,
                 synthetic_dir: str = None,
                 ):

        self.examples_per_class = examples_per_class
        self.generative_aug = generative_aug

        self.synthetic_probability = synthetic_probability
        self.synthetic_dir = synthetic_dir
        self.synthetic_examples = defaultdict(list)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5]),
        ])
        
        if synthetic_dir is not None:
            os.makedirs(synthetic_dir, exist_ok=True)
    
    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def generate_augmentations(self, num_repeats: int):

        self.synthetic_examples.clear()
        options = product(range(len(self)), range(num_repeats))

        for idx, num in tqdm(list(
                options), desc="Generating Augmentations"):

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

            image, label = self.generative_aug(
                image, label, self.get_metadata_by_idx(idx))

            if self.synthetic_dir is not None:

                pil_image, image = image, os.path.join(
                    self.synthetic_dir, f"aug-{idx}-{num}.png")

                pil_image.save(image)

            self.synthetic_examples[idx].append((image, label))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if len(self.synthetic_examples[idx]) > 0 and \
                np.random.uniform() < self.synthetic_probability:

            image, label = random.choice(self.synthetic_examples[idx])
            if isinstance(image, str): image = Image.open(image)

        else:

            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)

        return self.transform(image), label



class HugFewShotDataset(Dataset):

    num_classes: int = None
    class_names: int = None
    class2label: dict = None
    label2class: dict = None

    def __init__(self, 
                 split: str = "train",
                 examples_per_class: int = None,
                 synthetic_probability: float = 0.5,
                 synthetic_meta_type: str = 'csv',
                 return_onehot: bool = False,
                 soft_scaler: float = 1,
                 synthetic_dir: Union[str , List[str]] = None,
                 image_size: int = 512,
                 crop_size: int = 448,
                 soft_power: int = 1,
                 num_syn_seeds: int = 99999,
                 clip_filtered_syn: bool = False,
                 target_class_num: int = None,   
                 **kwargs):
      
        self.examples_per_class = examples_per_class
        self.num_syn_seeds = num_syn_seeds # number of seeds to generate synthetic data

        self.synthetic_dir = synthetic_dir
        self.clip_filtered_syn = clip_filtered_syn
        self.return_onehot = return_onehot


        if self.synthetic_dir is not None:
            assert self.return_onehot == True
            self.synthetic_probability = synthetic_probability
            self.soft_scaler = soft_scaler
            self.soft_power = soft_power
            self.target_class_num=target_class_num
            if synthetic_meta_type == 'csv':
                self.parse_syn_data_pd(synthetic_dir)
                self.get_syn_item = self.get_syn_item_pd
            else:
                self.parse_syn_data_pt(synthetic_dir)
                self.get_syn_item = self.get_syn_item_pt
                
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(crop_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        test_transform = transforms.Compose(
            [   
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.transform = {"train": train_transform, "val": test_transform}[split]

    def set_transform(self, transform) -> None:
        self.transform = transform

    @abc.abstractmethod
    def get_image_by_idx(self, idx: int) -> Image.Image:

        return NotImplemented
    
    @abc.abstractmethod
    def get_label_by_idx(self, idx: int) -> int:

        return NotImplemented
    
    @abc.abstractmethod
    def get_metadata_by_idx(self, idx: int) -> dict:

        return NotImplemented

    def parse_syn_data_pt(self, synthetic_dir, filter=True) -> None:
        meta_dir = os.path.join(synthetic_dir, "baseline_top3_logits.pt")
        self.meta_pt = torch.load(meta_dir)
        self.syn_nums = len(self.meta_pt['path'])    

    def get_syn_item_pt(self, idx: int):

        path = self.meta_pt['path'][idx]
        indices = self.meta_pt['topk_indices'][idx]
        logits = self.meta_pt['topk_logits'][idx]
        onehot_label = torch.zeros(self.num_classes)
        onehot_label[indices] =  torch.nn.functional.softmax(torch.tensor(logits),dim=0)
        onehot_label = self.soft_scaler * onehot_label

        return os.path.join(self.synthetic_dir,'data', path), onehot_label

    def parse_syn_data_pd(self, synthetic_dir, filter=True) -> None:
        if isinstance(synthetic_dir, list):
            pass
        elif isinstance(synthetic_dir, str):
            synthetic_dir = [synthetic_dir]
        else:
            raise NotImplementedError('Not supported type')
        meta_df_list = []
        for _dir in synthetic_dir:
            df_basename = "meta.csv" if not self.clip_filtered_syn else "remained_meta.csv"
            meta_dir = os.path.join(_dir, df_basename)
            meta_df = self.filter_df(pd.read_csv(meta_dir))
            meta_df.loc[:,'Path'] = meta_df['Path'].apply(lambda x: os.path.join(_dir,'data',x))
            meta_df_list.append(meta_df)
        self.meta_df = pd.concat(meta_df_list).reset_index(drop=True)
        self.syn_nums = len(self.meta_df)

        print(f"Syn numbers: {self.syn_nums}\n")
    
    def filter_df(self, df:pd.DataFrame) -> pd.DataFrame:

        if self.target_class_num is not None:
            selected_indexs=[]
            for source_name in self.class_names:
                target_classes = random.sample(self.class_names, self.target_class_num)
                indexs = df[(df['First Directory']==source_name) &(df['Second Directory'].isin(target_classes))]
                selected_indexs.append(indexs)

            meta2 = pd.concat(selected_indexs,axis=0)
            total_num = min(len(meta2),18000)
            idxs=random.sample(range(len(meta2)),total_num)
            meta2 = meta2.iloc[idxs]
            meta2.reset_index(drop=True,inplace=True)
            df = meta2
            print('filter_df',self.target_class_num,len(df))
        return df
    
    def get_syn_item_pd(self, idx: int):

        df_data = self.meta_df.iloc[idx]
        src_label = self.class2label[df_data['First Directory']]
        tar_label = self.class2label[df_data['Second Directory']]
        path = df_data['Path']
        strength = df_data['Strength']
        onehot_label = torch.zeros(self.num_classes)
        onehot_label[src_label] += self.soft_scaler * (1 - math.pow(strength,self.soft_power))
        onehot_label[tar_label] += self.soft_scaler * math.pow(strength,self.soft_power)

        return path, onehot_label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if  self.synthetic_dir is not None and \
                np.random.uniform() < self.synthetic_probability:
            syn_idx = np.random.choice(self.syn_nums)
            image, label = self.get_syn_item(syn_idx)
            if isinstance(image, str): image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)
        if self.return_onehot:
            if isinstance(label, (int, np.int64)): label = onehot(self.num_classes, label)
        return dict(pixel_values = self.transform(image),  label=label)