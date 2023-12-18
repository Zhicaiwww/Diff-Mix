import sys
import torch
import os
sys.path.append(os.getcwd())
import tqdm
import json
from PIL import Image
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from image_classification import CUSTOM_DATASET_BANK_V2

def main(args):
    checkpoint = args.checkpoint
    output_path = args.output_path
    dataset_name = args.dataset_name
    top_k = args.top_k
    meta_csv = args.meta_csv
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageClassification.from_pretrained(checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)

    train_transforms = Compose([
        RandomResizedCrop(crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])
    val_transforms = Compose([
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ])

    _, ds_test = CUSTOM_DATASET_BANK_V2[dataset_name]()
    ds_test.set_transform(val_transforms)

    # 设定GPU设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 设置batch size
    batch_size = 16

    root_path = os.path.join(os.path.dirname(meta_csv), 'data')
    meta_df = pd.read_csv(meta_csv)

    # meta_df = meta_df[(meta_df['First Directory'] == 'American Goldfinch') &
    #                   (meta_df['Second Directory'] == 'Shiny Cowbird') &
    #                   (meta_df['Number'] == 0)]

    # meta_df = meta_df.sort_values(by='Strength')
    img_paths = meta_df['Path'].values
    strengths = meta_df['Strength'].values
    targets = meta_df['Second Directory'].values

    # 初始化模型和转换函数（请替换成您实际使用的模型和转换函数）
    model = model.to(device)
    val_transforms = val_transforms  # 替换为您的图像转换函数

    result_dict = defaultdict(list)
    for batch_start in tqdm.tqdm(range(0, len(img_paths), batch_size)):
        batch_paths = img_paths[batch_start:batch_start+batch_size]
        batch_imgs = []

        for path in batch_paths:
            img = Image.open(os.path.join(root_path, path)).convert('RGB')
            img = val_transforms(img)
            batch_imgs.append(img)

        batch_imgs = torch.stack(batch_imgs).to(device)
        outputs = model(pixel_values=batch_imgs)

        topk_logits, topk_indices = torch.topk(outputs.logits.detach().cpu(), k=top_k, dim=-1)
        result_dict['path'].extend(batch_paths)
        result_dict['topk_logits'].append(np.array(topk_logits))
        result_dict['topk_indices'].append(np.array(topk_indices))

    result_dict['topk_logits'] = np.concatenate(result_dict['topk_logits'],axis=0)
    result_dict['topk_indices'] = np.concatenate(result_dict['topk_indices'],axis=0)
    result_dict['first Directory'].extend(meta_df['First Directory'].values)
    result_dict['second Directory'].extend(meta_df['Second Directory'].values)
    result_dict['strength'].append(strengths)
    result_dict['topk'] = top_k
    result_dict['root_dir'] = root_path
    # 将top3 logits和软标签信息保存为JSON文件

    torch.save(result_dict, f'{output_path}.pt')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='/data/zhicai/code/da-fusion/outputs/image_classification/202309020314-resnet50-tiny_cub', help='Path to the model checkpoint')
    parser.add_argument('--output_path', type=str, default='tiny_cub_baseline_top3_logits', help='Path to save the output JSON file')
    parser.add_argument('--dataset_name', type=str, default='tiny_cub', help='Name of the dataset')
    parser.add_argument('--meta_csv', type=str, default='/data/zhicai/code/da-fusion/outputs/aug_samples/tiny-bird-db_lora/meta.csv', help='Name of the dataset')
    parser.add_argument('--gpu-id', type=int, default=7, help='GPU ID to use')
    parser.add_argument('--top-k', type=int, default=3, help='GPU ID to use')

    args = parser.parse_args()

    main(args)
