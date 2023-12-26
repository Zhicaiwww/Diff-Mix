
import sys
import os
import yaml
import re
import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms 

from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from typing import Union, List, Tuple, Optional, Callable, Any, Dict, TypeVar, Generic

sys.path.append('/data/zhicai/code/Diff-Mix')
from semantic_aug.datasets.cub import CUBBirdHugDatasetForT2I, CUBBirdHugDataset, CUBBirdHugImbalanceDataset,CUBBirdHugImbalanceDatasetForT2I
from semantic_aug.datasets.flower import FlowersDatasetForT2I, Flowers102Dataset, FlowersImbalanceDataset, FlowersImbalanceDatasetForT2I
from semantic_aug.datasets.aircraft import AircraftHugDatasetForT2I, AircraftHugDataset
from semantic_aug.datasets.car import CarHugDatasetForT2I, CarHugDataset
from semantic_aug.datasets.chest import ChestHugDatasetForT2I, ChestHugDataset
from semantic_aug.datasets.pet import PetHugDatasetForT2I, PetHugDataset
from semantic_aug.datasets.food import FoodHugDatasetForT2I, FoodHugDataset
from semantic_aug.datasets.caltech101 import Caltech101DatasetForT2I, Caltech101Dataset
from semantic_aug.datasets.pascal import PascalDatasetForT2I, PascalDataset
from semantic_aug.datasets.dog import StanfordDogDatasetForT2I, StanfordDogDataset
from semantic_aug.datasets.pathmnist import PathMNISTDatasetForT2I, PathMNISTDataset

from semantic_aug.augmentations.textual_inversion import TextualInversionMixup
from semantic_aug.augmentations.dreabooth_lora import DreamboothLoraMixup, DreamboothLoraGeneration
from semantic_aug.augmentations.real_generation import RealGeneration

T2I_DATASET_NAME_MAPPING = {
                    'cub': CUBBirdHugDatasetForT2I,
                    'car': CarHugDatasetForT2I,
                    'aircraft': AircraftHugDatasetForT2I ,
                    'flower': FlowersDatasetForT2I ,
                    'chest': ChestHugDatasetForT2I,
                    'food': FoodHugDatasetForT2I,
                    'pet': PetHugDatasetForT2I,
                    'caltech': Caltech101DatasetForT2I,
                    'pascal': PascalDatasetForT2I,
                    'dog': StanfordDogDatasetForT2I,
                    'pathmnist': PathMNISTDatasetForT2I
                }

DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugDataset,
    "flower": Flowers102Dataset,
    "car": CarHugDataset,
    "chest": ChestHugDataset,
    "pet": PetHugDataset,
    "aircraft": AircraftHugDataset,
    "food": FoodHugDataset,
    "caltech": Caltech101Dataset,
    "pascal": PascalDataset,
    "dog": StanfordDogDataset,
    'pathmnist': PathMNISTDataset
}
IMBALANCE_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugImbalanceDataset,
    "flower": FlowersImbalanceDataset,
}
T2I_IMBALANCE_DATASET_NAME_MAPPING = {
    "cub": CUBBirdHugImbalanceDatasetForT2I,
    "flower": FlowersImbalanceDatasetForT2I,
}

AUGMENT = {
    "textual-inversion-mixup": TextualInversionMixup,
    "textual-inversion-augmentation": TextualInversionMixup,
    "real-guidance": DreamboothLoraMixup,
    "real-mixup": DreamboothLoraMixup,
    "real-generation": RealGeneration,
    "dreambooth-lora-mixup": DreamboothLoraMixup,
    "dreambooth-lora-augmentation": DreamboothLoraMixup,
    "dreambooth-lora-generation": DreamboothLoraGeneration,
}



def visualize_images(images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
                     nrow: int = 4,
                     show = False,
                     save = True,
                     outpath=None):

    if isinstance(images[0],Image.Image):
        transform = transforms.ToTensor()
        images_ts = torch.stack([transform(image) for image in images])
    elif isinstance(images[0],torch.Tensor):
        images_ts = torch.stack(images)
    elif isinstance(images[0],np.ndarray):
        images_ts = torch.stack([torch.from_numpy(image) for image in images])
    # save images to a grid
    grid = make_grid(images_ts, nrow=nrow, normalize=True, scale_each=True)
    # set plt figure size to (4,16)

    if show:
        plt.figure(figsize=(4*nrow,4 * (len(images) // nrow + (len(images) % nrow > 0))))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()
        # remove the axis
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    if save :
        assert outpath is not None
        if os.path.dirname(outpath) and not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath),exist_ok=True)
        img.save(f'{outpath}')
    return img 
 

 
def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def freeze_model(model, finetune_strategy='linear'):
    if finetune_strategy == 'linear':
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['layer4', 'fc']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages3-4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['layer3','layer4','fc']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages2-4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['layer2','layer3','layer4','fc']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages1-4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['layer1','layer2','layer3','layer4','fc']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'all':
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError(f'{finetune_strategy}') 
    
    trainable_params, total_params = count_parameters(model)
    ratio = trainable_params / total_params

    # print(f"Trainable Parameters: {trainable_params}")
    # print(f"Total Parameters: {total_params}")
    print(f"{finetune_strategy}, Trainable / Total Parameter Ratio: {ratio:.4f}")


def count_files_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count


def check_synthetic_dir_validity(synthetic_dir):
    if not os.path.exists(synthetic_dir):
        raise FileNotFoundError(f"Directory '{synthetic_dir}' does not exist.")

    total_files = count_files_in_directory(synthetic_dir)
    if total_files > 100:
        print(f"Directory '{synthetic_dir}' is valid with {total_files} files.")
    else:
        raise ValueError(f"Directory '{synthetic_dir}' contains less than 100 files, which is insufficient.")

def check_synthetic_dir_is_not_already(synthetic_dir):
    if not os.path.exists(synthetic_dir) or not os.path.exists(os.path.join(synthetic_dir, 'config.yaml')):
        print(f"Directory '{synthetic_dir}' does not exist.")
        return True
    else:
        total_files = count_files_in_directory(synthetic_dir)
        config = yaml.load(open(os.path.join(synthetic_dir, 'config.yaml')), Loader=yaml.BaseLoader)
        if total_files < int(config['total_tasks']):
            return True
        else:
            print(f"Directory '{synthetic_dir}' already exists with {total_files} files.")
            return False

    
def parse_synthetic_dir(dataset_name ,synthetic_type='mixup'): 
    synthetic_dir_meta_path = 'config/synthetic_datasets.yaml'
    import yaml
    synthetic_meta = yaml.load(open(synthetic_dir_meta_path), Loader=yaml.BaseLoader)
    if isinstance(synthetic_type, str):
        synthetic_dir = synthetic_meta[dataset_name][synthetic_type]
        check_synthetic_dir_validity(synthetic_dir)

    elif isinstance(synthetic_type, list):
        for syn_type in synthetic_type:
            synthetic_dir = synthetic_meta[dataset_name][syn_type] 
            check_synthetic_dir_validity(synthetic_dir)
    else:
        raise ValueError('synthetic_type should be str or list')
    print(f'{dataset_name}\t\t:{synthetic_type}' )

    return synthetic_dir

def parse_finetuned_ckpt(dataset, finetune_model_key='db_ti_latest'):
    with open('config/finetuned_ckpts.yaml', 'r') as file:
        import yaml
        meta_data = yaml.safe_load(file)
    lora_path = meta_data[dataset][finetune_model_key]['lora_path'] 
    embed_path = meta_data[dataset][finetune_model_key]['embed_path'] 
    return lora_path, embed_path

def checked_has_run(exp_dir, args):
    import copy
    parent_dir = os.path.abspath(os.path.join(exp_dir, os.pardir))
    current_args = copy.deepcopy(args)
    current_args.pop('gpu' , None)
    current_args.pop('note', None)
    current_args.pop('target_class_num',None)
    
  
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for dirname in dirnames:
            config_file = os.path.join(dirpath, dirname, 'config.yaml')
            if os.path.exists(config_file):
                with open(config_file, 'r') as file:
                    saved_args = yaml.load(file, Loader=yaml.FullLoader)
                
                if current_args['syn_type'] is None or 'aug' in current_args['syn_type'] or 'gen' in current_args['syn_type']:
                    current_args.pop('soft_power',None)
                    saved_args.pop('soft_power', None)
                saved_args.pop('gpu',  None)
                saved_args.pop('note', None)
                saved_args.pop('target_class_num',None)
                if saved_args == current_args:
                    print(f'This program has already been run in directory: {dirpath}/{dirname}')
                    return True
    return False


def parse_result(target_dir, extra_column=[], postfix='_5shot'):
    results=[]
    for file in os.listdir(target_dir):
        config_file = os.path.join(target_dir,file,'config.yaml')
        config = yaml.safe_load(open(config_file,'r'))
        if isinstance(config['syn_type'],list):
            syn_type = config['syn_type'][0]
        else:
            syn_type = config['syn_type']
            
        if syn_type is None:
            strategy='baseline'
            strength=0
        else:
            match = re.match(r'([a-zA-Z]+)([0-9.]*).*', syn_type)
            if match:
                strategy = match.group(1)
                strength = match.group(2)
            else:
                continue
        for basefile in os.listdir(os.path.join(target_dir,file)):
            if 'acc_eval' in basefile:
                acc = float(basefile.split('_')[-1])
                results.append((config['dir'],config['res_mode'],config['lr'], strategy, strength, config['soft_power'],config['seed'],*[str(config.pop(key, 'False')) for key in extra_column], acc ))
                break

    df = pd.DataFrame(results, columns=['dataset','resolution', 'lr','strategy', 'strength','soft power', 'seed', *extra_column, 'acc'])
    df['acc'] = df['acc'].astype(float)
    result_seed = df.groupby(['dataset','resolution', 'lr' ,'strength', 'strategy', 'soft power', *extra_column]).agg({'acc': ['mean','var']}).reset_index()
    result_sorted = result_seed.sort_values(by=['dataset','resolution', 'lr' ,'strategy', 'strength',*extra_column])
    result_seed.columns = ['_'.join(col).strip() for col in result_seed.columns.values]

    return result_sorted


if __name__ == '__main__':
    for name in ['ti_mixup', 'db_mixup', 'db_ti_mixup', 'mixup_s5000', 'mixup_s15000', 'mixup_s25000', 'mixup_s35000', 'aug_s5000', 'aug_s15000', 'aug_s25000', 'aug_s35000', 'mixup_uniform120000']:
        parse_synthetic_dir('cub', name)