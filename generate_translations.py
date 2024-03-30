
from typing import Any
import os
import sys
import re
import torch
import argparse
import random
import time
import numpy as np
import pandas as pd
import math
sys.path.append('/data/zhicai/code/Diff-Mix/')
os.environ['CURL_CA_BUNDLE'] = ''

from utils import DATASET_NAME_MAPPING,IMBALANCE_DATASET_NAME_MAPPING, AUGMENT_METHODS, parse_finetuned_ckpt
from tqdm import tqdm
from PIL import Image
from itertools import product
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty



def check_args_videlity(args):
        
    if args.sample_strategy == 'real-gen':
        args.lora_path = None
        args.embed_path = None
        args.aug_strength = 1
        output_name = f'{args.sample_strategy}-Multi{args.syn_dataset_mulitiplier}'
    elif args.sample_strategy == 'diff-gen':
        lora_path, embed_path = parse_finetuned_ckpt(args.dataset, args.finetune_model_key)
        args.lora_path = lora_path
        args.embed_path = embed_path 
        output_name = f'{args.sample_strategy}-Multi{args.syn_dataset_mulitiplier}_{args.finetune_model_key}'
        args.aug_strength = 1
    else:
        if args.sample_strategy in ['real-guidance','real-mix']:
            args.lora_path = None
            args.embed_path = None
            output_name = f'{args.sample_strategy}-Multi{args.syn_dataset_mulitiplier}'
        else:
            lora_path, embed_path = parse_finetuned_ckpt(args.dataset, args.finetune_model_key)
            args.lora_path = lora_path 
            args.embed_path = embed_path 
            output_name = f'{args.sample_strategy}-Multi{args.syn_dataset_mulitiplier}-{args.finetune_model_key}'

        if args.mask:
            output_name += '-Mask'
        if args.strength_strategy == 'fixed':
            assert args.aug_strength is not None
            output_name += f'-Strength{args.aug_strength}'
        elif args.strength_strategy == 'beta':
            assert args.beta_strength is not None
            output_name += f'-Beta{np.round(args.beta_strength,1)}'
        elif args.strength_strategy == 'uniform':
            output_name += f'-Uniform'
    return output_name

def beta_sampling(num_samples, beta=5):

    if num_samples < 1:
        raise ValueError("Number of samples must be greater than or equal to 1")
    # beta distribution with alpha = 1, beta = 1, clamp between 0.1 to 0.9
    min_value = 0.1
    max_value = 0.9

    # Generate and clamp the values
    sampled_values = [np.clip(np.round(np.random.beta(beta, beta), 3),a_min=min_value,a_max=max_value) for _ in range(num_samples)]
    
    return sampled_values
 
def generate_translations(args, in_queue, out_queue, gpu_id, process_id):
    # print(f'seed: {args.seed + process_id}')

    os.environ['CURL_CA_BUNDLE'] = ''
    
    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)
    if args.task == 'imbalanced':
        train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, resolution=args.resolution, imbalance_factor=args.imbalance_factor)
    else:
        train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, examples_per_class=args.examples_per_class, resolution=args.resolution, corrupt_prob=args.corrupt_prob)
    model = AUGMENT_METHODS[args.sample_strategy](
    model_path = args.model_path,
    embed_path=args.embed_path,
    lora_path=args.lora_path, 
    prompt=args.prompt, 
    guidance_scale=args.guidance_scale,
    mask=args.mask, 
    inverted=args.inverted,
    device=f'cuda:{gpu_id}'
    )
    batch_size = args.batch_size 
    while True:
        index_list = []
        source_label_list = []
        target_label_list = []
        strength_list = []
        for _ in range(batch_size):
            try:
                index, source_label, target_label, strength = in_queue.get(timeout=1)
                index_list.append(index)
                source_label_list.append(source_label)
                target_label_list.append(target_label)
                strength_list.append(strength)
                # print(out_queue.qsize())
            except Empty:
                print("queue empty, exit")
                break
            # strength = np.round(np.random.beta_strength(args.beta_strength, args.beta_strength),2)
        target_label=target_label_list[0]
        target_indice = random.sample(train_dataset.label_to_indices[target_label], 1)[0]
        target_metadata = train_dataset.get_metadata_by_idx(target_indice)
        target_name = target_metadata['name'].replace(" ", "_").replace("/", "_")

        source_images = []
        save_paths =[]
        if args.task == 'vanilla':
            source_indices = [random.sample(train_dataset.label_to_indices[source_label], 1)[0] for source_label in source_label_list]
        elif args.task == 'imbalanced':
            source_indices = random.sample(range(len(train_dataset)), batch_size)
        for index, source_indice in zip(index_list, source_indices):
            source_images.append(train_dataset.get_image_by_idx(source_indice))
            source_metadata = train_dataset.get_metadata_by_idx(source_indice)
            source_name = source_metadata['name'].replace(" ", "_").replace("/", "_")
            save_name = os.path.join(source_name, f"{target_name}-{index:06d}-{strength}.png")
            save_paths.append(os.path.join(args.output_path, 'data', save_name))

        if os.path.exists(save_paths[0]):
            print(f"skip {save_paths[0]}")
        else:
            image, _ = model(image=source_images, label=target_label, strength=strength, metadata=target_metadata,
                             resolution=args.resolution)
            for image, save_path in zip(image, save_paths):
                image.save(save_path)
            print(f"save {save_path}")

    print("I am Done ! ! ! ")
    
def main(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gpu_ids = args.gpu_ids
    in_queue = Queue()
    out_queue = Queue()

#region Configure sampling options
    if args.task == 'imbalanced':
        train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, resolution=args.resolution, imbalance_factor=args.imbalance_factor)
    else:
        train_dataset = DATASET_NAME_MAPPING[args.dataset](split="train", seed=args.seed, examples_per_class=args.examples_per_class, resolution=args.resolution)

    num_classes = len(train_dataset.class_names)
    for name in train_dataset.class_names:
        name = name.replace(' ','_').replace("/", "_")
        os.makedirs(os.path.join(args.output_path, 'data' ,name), exist_ok=True)

    num_tasks = args.syn_dataset_mulitiplier * len(train_dataset)

    if args.sample_strategy in ['real-gen','real-guidance', 'diff-aug','diff-gen', 'textual-inversion-augmentation']:
        source_classes = random.choices(range(len(train_dataset.class_names)), k=num_tasks)
        target_classes = source_classes
    elif args.sample_strategy in ['real-mix', 'diff-mix', 'textual-inversion-mixup']:
        source_classes = random.choices(range(len(train_dataset.class_names)), k=num_tasks)
        target_classes = random.choices(range(len(train_dataset.class_names)), k=num_tasks)
    else:
        raise ValueError(f"Augmentation strategy {args.sample_strategy} not supported")

    if args.strength_strategy == 'fixed':
        strength_list = [args.aug_strength] * num_tasks
    elif args.strength_strategy == 'beta':
        strength_list = beta_sampling(num_tasks,args.beta_strength)
    elif args.strength_strategy == 'uniform':
        strength_list = random.choices([0.3, 0.5, 0.7, 0.9], k=num_tasks)
    options = zip(range(num_tasks), source_classes, target_classes, strength_list)

    for option in options:
        in_queue.put(option)

        
    sample_config_path = os.path.join(args.output_path, 'config.yaml')
    sample_config = vars(args)
    sample_config['num_classes'] = num_classes
    sample_config['total_tasks'] = num_tasks


    from utils import check_synthetic_dir_is_not_already
    if check_synthetic_dir_is_not_already(args.output_path):
        pass
    else:
        exit()
    sample_config['strength_strategy'] = args.strength_strategy
    
    with open(sample_config_path, 'w') as f:
        import yaml
        yaml.dump(sample_config, f)
#endregion

    processes = []
    total_tasks = in_queue.qsize()
    print('Number of total tasks', total_tasks)

    with tqdm(total=total_tasks, desc="Processing") as pbar:
        for process_id, gpu_id in enumerate(gpu_ids):
            process = Process(target=generate_translations, args=(args, in_queue, out_queue, gpu_id, process_id))
            process.start()
            processes.append(process)

        while any(process.is_alive() for process in processes):
            current_queue_size = in_queue.qsize()
            pbar.n = total_tasks - current_queue_size
            pbar.refresh()
            time.sleep(1)  

        for process in processes:
            process.join()

    rootdir = os.path.join(args.output_path,'data')
    pattern_level_1 = r"(.+)"
    pattern_level_2 = r"(.+)-(\d+)-(.+).png"

    # Generate meta.csv for indexing images
    data_dict = defaultdict(list)
    for dir in os.listdir(rootdir):
        if not os.path.isdir(os.path.join(rootdir, dir)):
            continue
        match_1 = re.match(pattern_level_1, dir)
        first_dir = match_1.group(1).replace("_", " ") 
        for file in os.listdir(os.path.join(rootdir, dir)):
            match_2 = re.match(pattern_level_2, file)
            second_dir = match_2.group(1).replace("_", " ")
            num = int(match_2.group(2))
            floating_num = float(match_2.group(3))
            data_dict['First Directory'].append(first_dir)
            data_dict['Second Directory'].append(second_dir)
            data_dict['Number'].append(num)
            data_dict['Strength'].append(floating_num)
            data_dict['Path'].append(os.path.join(dir, file))

    df = pd.DataFrame(data_dict)

    # Validate generated images
    valid_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.output_path,'data',row['Path'])
        try:
            img = Image.open(image_path)
            img.close()  
            valid_rows.append(row)
        except Exception as e:
            os.remove(image_path)
            print(f"Deleted {image_path} due to error: {str(e)}")

    valid_df = pd.DataFrame(valid_rows)
    csv_path = os.path.join(args.output_path, 'meta.csv')
    valid_df.to_csv(csv_path, index=False)

    print("DataFrame:")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument("--output-root", type=str, default='outputs/aug_samples')
    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--confusion-matrix-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="pascal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-class", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    parser.add_argument("--sample_strategy", type=str, default="textual-inversion-mixup", 
                        choices=["real-gen","real-guidance","real-mix", "textual-inversion-augmentation" ,"textual-inversion-mixup", "diff-aug","diff-mix","diff-gen"])
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--mask", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inverted", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpu-ids", type=int, nargs='+', default=[0])
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--task", type=str, default='vanilla', choices=['vanilla', 'imbalanced'])
    parser.add_argument("--imbalance_factor", type=float, default=0.01, choices=[0.01,0.05,0.1])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--syn_dataset_mulitiplier", type = int, default = 5, help = "multiplier for the number of synthetic images compared to the number of real images")

    parser.add_argument("--strength_strategy", type=str, default='fixed', choices=['fixed','beta','uniform'])
    parser.add_argument("--finetune_model_key", type=str, default='')
    parser.add_argument("--beta_strength", type = int, default = None)
    parser.add_argument("--aug_strength", type = float, default = None)
    parser.add_argument("--corrupt_prob", type = float, default = 0)

    args = parser.parse_args()

    # Please check the corresponding ckpt path before sampling !!! 

    torch.multiprocessing.set_start_method('spawn')
    os.makedirs(os.path.join(args.output_root, args.dataset), exist_ok=True)
    output_name = check_args_videlity(args)
    args.output_path = os.path.join(args.output_root, args.dataset ,output_name)
    
    main(args)

