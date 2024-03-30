import sys
import os
import random
import torch
sys.path.append('/data/zhicai/code/Diff-Mix/')
from semantic_aug.few_shot_dataset import FewShotDataset, HugFewShotDataset
from semantic_aug.datasets.utils import IMAGENET_TEMPLATES_SMALL 
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from shutil import copyfile
from PIL import Image, ImageDraw
from collections import defaultdict
from datasets import load_dataset, load_from_disk


SUPER_CLASS_NAME = 'bird'
IMAGE_DIR = r"/data/zhicai/datasets/fgvc_datasets/CUB_200_2011"
HUG_LOCAL_IMAGE_TRAIN_DIR = "/data/zhicai/cache/huggingface/datasets/Multimodal-Fatima___parquet/Multimodal-Fatima--CUB_train-bc20d158956ded0c"
HUG_LOCAL_IMAGE_TEST_DIR = "/data/zhicai/cache/huggingface/datasets/Multimodal-Fatima___parquet/Multimodal-Fatima--CUB_test-71768dd1fcbe387e"


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def generate_mask(bbox, image_size):
    x, y, width, height = bbox
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((x, y, x + width, y + height), fill=255)
    
    return mask

class CUBBirdDataset(HugFewShotDataset):
    super_class_name = SUPER_CLASS_NAME
    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_dir: str = IMAGE_DIR, 
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

        data_dir = image_dir
        self.data_dir = data_dir
        filepath = os.path.join(data_dir, 'images.txt')
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        train_test_split_path = os.path.join(data_dir, 'train_test_split.txt')
        image_class_labels_path = os.path.join(data_dir, 'image_class_labels.txt')  
        classes_path = os.path.join(data_dir, 'classes.txt')
        df_bbox = pd.read_csv(bbox_path, delim_whitespace=True, header=None, names=['x', 'y', 'width', 'height'], index_col=0)
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None, names=['filepath'], index_col=0)
        df_train_test_split = pd.read_csv(train_test_split_path, delim_whitespace=True, header=None, names=['train_test_split'], index_col=0)
        df_image_class_labels = pd.read_csv(image_class_labels_path, delim_whitespace=True, header=None, names=['class_id'], index_col=0)
        df_image_class_labels = df_image_class_labels - 1
        #
        df_dataset_meta = pd.concat([df_filenames, df_train_test_split, df_image_class_labels, df_bbox], axis=1)

        if split == 'train':
            df_dataset_meta = df_dataset_meta[df_dataset_meta['train_test_split'] == 1]
        else:
            df_dataset_meta = df_dataset_meta[df_dataset_meta['train_test_split'] == 0]

        classes_df = pd.read_csv(classes_path, sep=' ', header=None, names=['id', 'name'])
        classes_df['name'] = classes_df['name'].str[4:].str.replace('_', ' ')
        classes_df['id'] = classes_df['id'] - 1
        self.label2class = classes_df.set_index('id').to_dict()['name']
        self.class2label = {v: k for k, v in self.label2class.items()}
        self.class_names = list(self.class2label.keys())
        self.num_classes = len(self.class_names)

        random.seed(seed)
        np.random.seed(seed)
        # shuffle the dataset
        df_dataset_meta = df_dataset_meta.sample(frac=1).reset_index(drop=True) 
        # TODO examples_per_class
        
        self.label_to_indices = defaultdict(list)
        for i in range(len(df_dataset_meta)):
            self.label_to_indices[df_dataset_meta.iloc[i]['class_id']].append(i)

        self.df_dataset_meta = df_dataset_meta

    def __len__(self):
        
        return len(self.df_dataset_meta)

    def get_image_by_idx(self, idx: int) -> Image.Image:
        
        image_path = os.path.join(self.data_dir, 'images', self.df_dataset_meta.iloc[idx]['filepath'])
        image = Image.open(image_path).convert('RGB')
        return image

    def get_label_by_idx(self, idx: int) -> int:

        return  self.df_dataset_meta.iloc[idx]['class_id']

    def get_metadata_by_idx(self, idx: int) -> dict:
        bbox = self.df_dataset_meta.iloc[idx][['x', 'y', 'width', 'height']].values
        image = self.get_image_by_idx(idx)
        mask = generate_mask(bbox, image.size)
        return dict(name=self.label2class[self.get_label_by_idx(idx)], mask = mask, super_class = self.super_class_name)

class CUBBirdHugDataset(HugFewShotDataset):
    super_class_name = SUPER_CLASS_NAME
    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 image_test_dir: str = HUG_LOCAL_IMAGE_TEST_DIR, 
                 examples_per_class: int = -1, 
                 synthetic_probability: float = 0.5,
                 return_onehot: bool = False,
                 soft_scaler: float = 0.9,
                 synthetic_dir: str = None,
                 image_size: int = 512,
                 crop_size: int = 448,
                 corrupt_prob=0.0,
                 **kwargs):

        if split == 'train':
            dataset = load_dataset(HUG_LOCAL_IMAGE_TRAIN_DIR, split='train')
        else:
            dataset = load_dataset(HUG_LOCAL_IMAGE_TEST_DIR, split='test')

        random.seed(seed)
        np.random.seed(seed)
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
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.num_classes = len(self.class_names)
        if corrupt_prob > 0:
            print(f"corrupt_prob: {corrupt_prob}")
            self.corrupt_labels(corrupt_prob)
        class2label = self.dataset.features['label']._str2int
        self.class2label = {k.replace('/',' '): v for k, v in class2label.items()}
        self.label2class = {v: k.replace('/',' ') for k, v in class2label.items()} 

        
        
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)

        super().__init__(
            *args,split=split, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot, soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,image_size=image_size, crop_size=crop_size, **kwargs)    
        

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.dataset['label'])
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(len(self.class_names), mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]
        self.dataset = self.dataset.remove_columns("label").add_column("label", labels).cast(self.dataset.features)
        
    def __len__(self):
        
        return len(self.dataset)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return self.dataset[idx]['image'].convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.dataset[idx]['label']
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.label2class[self.get_label_by_idx(idx)], super_class = self.super_class_name)

class CUBBirdHugDatasetForT2I(torch.utils.data.Dataset):

    super_class_name = SUPER_CLASS_NAME

    def __init__(self, *args, split: str = "train",
                 seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 max_train_samples: int = -1,
                 class_prompts_ratio: float = 0.5,
                 resolution: int = 512,
                 center_crop: bool = False,
                 random_flip: bool = False,
                 use_placeholder: bool = False,
                 examples_per_class: int = -1,
                 **kwargs):

        super().__init__()    

        dataset = load_dataset(image_train_dir, split='train')

        random.seed(seed)
        np.random.seed(seed)
        if max_train_samples is not None and max_train_samples > 0:
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
        self.class_names = [name.replace('/',' ') for name in dataset.features['label'].names]
        self.dataset = dataset
        self.class2label = self.dataset.features['label']._str2int
        self.label2class = {v: k for k, v in self.class2label.items()} 
        self.num_classes = len(self.class_names)
        self.class_prompts_ratio = class_prompts_ratio
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
        ])

    

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
        flag = random.random() < self.class_prompts_ratio
        if flag:
            if self.use_placeholder:
                content = self.name2placeholder[self.label2class[self.dataset[idx]['label']]] + f' {self.super_class_name}'
            else:
                content = self.label2class[self.dataset[idx]['label']]
            prompt =  random.choice(IMAGENET_TEMPLATES_SMALL).format(content)
        else:
            prompt = random.choice(self.dataset[idx]['description'].strip().split('\n'))
        return prompt

    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.label2class[self.get_label_by_idx(idx)])

class CUBBirdHugImbalanceDataset(CUBBirdHugDataset):
    super_class_name = SUPER_CLASS_NAME
    def __init__(self, *args, split: str = "train", seed: int = 0, 
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR, 
                 image_test_dir: str = HUG_LOCAL_IMAGE_TEST_DIR, 
                 examples_per_class: int = -1, 
                 synthetic_probability: float = 0.5,
                 return_onehot: bool = False,
                 soft_scaler: float = 1.0,
                 synthetic_dir: str = None,
                 image_size: int = 512,
                 crop_size: int = 448,
                 use_randaug: bool = False,
                 imbalance_factor: float = 0.01,
                 weighted_alpha: int = 1,
                 use_weighted_syn: bool =False,
                 **kwargs):

        super().__init__(
            *args,split=split, examples_per_class=examples_per_class,
            synthetic_probability=synthetic_probability,
            return_onehot=return_onehot, soft_scaler=soft_scaler,
            synthetic_dir=synthetic_dir,image_size=image_size, crop_size=crop_size, **kwargs)    

        self.use_randaug = use_randaug
        self.weighted_alpha = weighted_alpha
        if split == 'train':
            self.gen_imbalanced_data(imbalance_factor)
            self.normalized_probabilities = self.weighted_prob() if use_weighted_syn else None
            if synthetic_dir is not None:
                self.syn_label_to_indices = self.create_bird_indices_dict()
        
    def gen_imbalanced_data(self, imb_factor):
        # sort dataset from most to least common class
        img_average = len(self.dataset) / self.num_classes
        org_num = len(self.dataset)
        label_to_indices = defaultdict(list)
        all_indices = []
        for sorted_idx, (sorted_label, indices) in  enumerate(sorted(self.label_to_indices.items(), key=lambda x: len(x[1]), reverse=True)):
            num_imgs_cur_label = len(indices)
            num = img_average * (imb_factor ** (sorted_idx / (self.num_classes - 2.0)))
            unbalance_indices = random.sample(indices, max(int(num),1))
            label_to_indices[sorted_label] = unbalance_indices
            all_indices.extend(unbalance_indices)
        self.dataset = self.dataset.select(all_indices)
        self.label_to_indices = defaultdict(list)
        for i, label in enumerate(self.dataset['label']):
            self.label_to_indices[label].append(i)
        cur_num = len(self.dataset)
        print(f'Dataset size filtered from {org_num} to {cur_num} with imbalance factor {imb_factor}' )

    def get_cls_num_list(self):
        cls_num_list = [len(self.label_to_indices[label]) for label in range(self.num_classes)]
        return cls_num_list

    def weighted_prob(self):
        cls_num_list = self.get_cls_num_list()
        probabilities = [1 / (num + 1) for num in cls_num_list]
        total_prob = sum(probabilities)
        normalized_probabilities = [prob / total_prob for prob in probabilities]
        print("\nUsing weighted probability ! \n")
        return normalized_probabilities 

    def create_bird_indices_dict(self):
        bird_indices = {}
        arr = self.meta_df['Second Directory'].values
        for label, bird_name  in self.label2class.items():
            indices = np.where(arr == bird_name)[0]
            bird_indices[label] = indices
        return bird_indices

    def get_weighted_sampler(self):
        
        cls_num_list = self.get_cls_num_list()
        print(cls_num_list)
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t['label']] for t in self.dataset])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset), replacement=True)
        return sampler

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        if  self.synthetic_dir is not None and \
                np.random.uniform() < self.synthetic_probability:

            if self.normalized_probabilities is not None:
                cls = np.random.choice(self.num_classes, p=self.normalized_probabilities)
                syn_idx = random.choice(self.syn_label_to_indices[cls])
                image, label = self.get_syn_item(syn_idx)
            else:
                syn_idx = np.random.choice(self.syn_nums)
                image, label = self.get_syn_item(syn_idx)
            if isinstance(image, str): image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)
            label = self.get_label_by_idx(idx)
        if self.return_onehot:
            if isinstance(label, (int, np.int64)): label = onehot(self.num_classes, label)

        if self.use_randaug:
            r = random.random()
            if r < 0.5:
                image = self.transform[0](image)
            else:
                image = self.transform[1](image)
        else:
            if self.transform is not None:
                image = self.transform(image)
        return dict(pixel_values = image,  label=label)

class CUBBirdHugImbalanceDatasetForT2I(CUBBirdHugDatasetForT2I):
    def __init__(self, *args, split: str = "train",
                 seed: int = 0,
                 image_train_dir: str = HUG_LOCAL_IMAGE_TRAIN_DIR,
                 max_train_samples: int = -1,
                 class_prompts_ratio: float = 0.5,
                 resolution: int = 512,
                 center_crop: bool = False,
                 random_flip: bool = False,
                 use_placeholder: bool = False,
                 examples_per_class: int = -1,
                 imbalance_factor: float = 0.01,
                 weighted_alpha: int = 1,
                 **kwargs):
        super().__init__(*args, split=split, seed=seed, image_train_dir=image_train_dir, max_train_samples=max_train_samples, 
                         class_prompts_ratio=class_prompts_ratio, resolution=resolution, center_crop=center_crop, 
                         random_flip=random_flip, use_placeholder=use_placeholder, examples_per_class=examples_per_class,
                         **kwargs)
        

        self.weighted_alpha = weighted_alpha
        self.gen_imbalanced_data(imbalance_factor)


    
    def gen_imbalanced_data(self, imb_factor):
        # sort dataset from most to least common class
        img_average = len(self.dataset) / self.num_classes
        org_num = len(self.dataset)
        label_to_indices = defaultdict(list)
        all_indices = []
        for sorted_idx, (sorted_label, indices) in  enumerate(sorted(self.label_to_indices.items(), key=lambda x: len(x[1]), reverse=True)):
            num_imgs_cur_label = len(indices)
            num = img_average * (imb_factor ** (sorted_idx / (self.num_classes - 2.0)))
            unbalance_indices = random.sample(indices, max(int(num),1))
            label_to_indices[sorted_label] = unbalance_indices
            all_indices.extend(unbalance_indices)
        self.dataset = self.dataset.select(all_indices)
        self.label_to_indices = label_to_indices
        cur_num = len(self.dataset)
        print(f'Dataset size filtered from {org_num} to {cur_num} with imbalance factor {imb_factor}' )

    def get_cls_num_list(self):
        cls_num_list = [len(self.label_to_indices[label]) for label in range(self.num_classes)]
        return cls_num_list
     
    def weighted_prob(self):
        cls_num_list = self.get_cls_num_list()
        probabilities = [1 / (num + 1) for num in cls_num_list]
        total_prob = sum(probabilities)
        normalized_probabilities = [prob / total_prob for prob in probabilities]
        print("\nUsing weighted probability ! \n")
        return normalized_probabilities 

    def get_weighted_sampler(self):
        
        cls_num_list = self.get_cls_num_list()
        print(cls_num_list)
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t['label']] for t in self.dataset])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset), replacement=True)
        return sampler  

class WaterBird(Dataset):
    def __init__(self, split=2, image_size=256,crop_size=224,return_onehot = False):
        self.root_dir = '/data/zhicai/datasets/waterbird_complete95_forest2water2/'
        dataframe = pd.read_csv(os.path.join(self.root_dir,'metadata.csv'))
        dataframe = dataframe[dataframe['split']==split].reset_index()
        self.labels = list(map(lambda x: int(x.split('.')[0]) -1, dataframe['img_filename']))
        self.dataframe = dataframe
        self.image_paths = dataframe['img_filename']
        self.groups = dataframe.apply(lambda row: f"{row['y']}{row['place']}", axis=1).tolist()
        self.return_onehot = return_onehot
        self.num_classes = len(set(self.labels))
        self.transform = transforms.Compose(
            [   
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ) 

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        # Load image
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')

        label = self.labels[idx]
        group = self.groups[idx]
        if self.transform:
            img = self.transform(img)
        if self.return_onehot:
            if isinstance(label, (int, np.int64)): label = onehot(self.num_classes, label)
        return img, label, group
    
if __name__== '__main__':
    ds = CUBBirdHugDataset(examples_per_class=5)
    save_dir = '/data/zhicai/code/Diff-Mix/data/5shot'
    import tqdm
    for i in tqdm.tqdm(range(len(ds))):
        image = ds.get_image_by_idx(i)
        image.save(os.path.join(save_dir,f'{i}.png'))
    