import requests
import tqdm
import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from semantic_aug.few_shot_dataset import SyntheticDataset
from utils import parse_synthetic_dir, DATASET_NAME_MAPPING


def to_tensor(x):
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError
def syn_collate_fn(examples):
    pixel_values = [example['pixel_values'] for example in examples]
    src_labels = torch.stack([to_tensor(example['src_label']) for example in examples])
    tar_labels = torch.stack([to_tensor(example['tar_label']) for example in examples])
    dtype = torch.float32  if len(src_labels.size()) == 2 else torch.long
    src_labels.to(dtype=dtype)
    tar_labels.to(dtype=dtype)
    return {'pixel_values': pixel_values, 'src_labels':src_labels, 'tar_labels':tar_labels}
def collate_fn(examples):
    pixel_values = [example['pixel_values'] for example in examples]
    labels = torch.stack([to_tensor(example['label']) for example in examples])
    dtype = torch.float32  if len(labels.size()) == 2 else torch.long
    labels.to(dtype=dtype)
    return {'pixel_values': pixel_values, 'labels':labels}
def main(args):
    device="cuda:6"
    threshold=0.6
    num_workers=16
    bs=64
    synthetic_dir = parse_synthetic_dir('cub', synthetic_type=args.synthetic_type)

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    ds = DATASET_NAME_MAPPING['cub'](split='train', return_onthot=False)
    ds.transform =  torch.nn.Identity()
    ds_syn = SyntheticDataset(synthetic_dir, class2label = ds.class2label)
    ds_syn.transform = torch.nn.Identity()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    dataloader_syn = torch.utils.data.DataLoader(ds_syn, batch_size=bs,collate_fn=syn_collate_fn, shuffle=False, num_workers=num_workers)

    ds_features = []
    ds_labels = []
    # with torch.no_grad():
    #     for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
    #         images = batch['pixel_values']
    #         labels = batch['labels'].to(device)
    #         inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    #         print(inputs.pixel_values.mean())
    #         output_features = model.vision_model(inputs.pixel_values).pooler_output
    #         ds_features.append(output_features)
    #         ds_labels.append(labels)

    #     ds_features = torch.cat(ds_features, dim=0).detach().cpu()
    #     ds_labels = torch.cat(ds_labels, dim=0).detach().cpu()
    #     prototype_feature_per_class = []
    #     for label in range(ds.num_classes):
    #         prototype_feature_per_class.append(ds_features[ds_labels==label].mean(dim=0))
    #     prototype_feature_per_class = torch.stack(prototype_feature_per_class)
    #     save_dict = {"features": ds_features,
    #                 "labels": ds_labels,
    #                 "prototype_feature_per_class": prototype_feature_per_class}
    #     torch.save(save_dict, 'cub_train_features.pt')
    
    prototype_feature_per_class = torch.load('cub_train_features.pt')['prototype_feature_per_class']
    
    # extract synthetic features
    ds_syn_features = []
    ds_syn_src_labels = []
    ds_syn_tar_labels = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader_syn, total=len(dataloader_syn)):
            images = batch['pixel_values']
            src_labels = batch['src_labels'].to(device)
            tar_labels = batch['tar_labels'].to(device)

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            print(inputs.pixel_values.mean())
            output_features = model.vision_model(inputs.pixel_values).pooler_output
            ds_syn_features.append(output_features)
            ds_syn_src_labels.append(src_labels)
            ds_syn_tar_labels.append(tar_labels)

        ds_syn_features = torch.cat(ds_syn_features, dim=0).detach().cpu()
        ds_syn_src_labels = torch.cat(ds_syn_src_labels, dim=0).detach().cpu()
        ds_syn_tar_labels = torch.cat(ds_syn_tar_labels, dim=0).detach().cpu()
        src_prototype_feature = prototype_feature_per_class[ds_syn_src_labels]
        tar_prototype_feature = prototype_feature_per_class[ds_syn_tar_labels]
        src_tar_delta_feature = tar_prototype_feature - src_prototype_feature 
        src_syn_delta_feature = ds_syn_features - src_prototype_feature
        tar_syn_delta_feature = ds_syn_features - tar_prototype_feature
        # compute the similarity
        src_distance = torch.norm(src_syn_delta_feature, dim=1) 
        tar_distance = torch.norm(tar_syn_delta_feature, dim=1)
        similarity = torch.nn.functional.softmax(torch.stack([src_distance, tar_distance], dim=1), dim=1)[:,1]
        similarity2 = tar_distance / (src_distance + tar_distance)
        print(similarity)
        print(similarity[:,0].mean())
        print(similarity2)
        print(similarity2.mean())
    ds_syn.meta_df['similarity'] = similarity
    ds_syn.meta_df.to_csv(os.path.join(synthetic_dir, 'meta_similarity.csv'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_type', type=str, default='mixup0.7')
    args = parser.parse_args()
    main(args) 