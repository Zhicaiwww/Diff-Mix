import requests
import tqdm
import torch
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from semantic_aug.few_shot_dataset import SyntheticDataset
from utils import parse_synthetic_dir

FILTER_CAPTIONS_MAPPING = {
'cub':[
    "a photo with a bird on it",
    "a photo without a bird on it"
],
'aircraft':[
    "a photo with an aricraft on it. ",
    "a photo without an aricraft on it. ",
],
'dog':[
    "a photo with a dog on it. ",
    "a photo without a dog on it. ",
],
'flower':[
    "a photo with a flower on it. ",
    "a photo without a flower on it. ",
],
'car':[
    "a photo with a car on it. ",
    "a photo without an car on it. ",
]
}
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
def main(args):
    device = f"cuda:{args.gpu}"
    bs = args.batch_size
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    synthetic_dir = parse_synthetic_dir(args.dataset, synthetic_type=args.synthetic_type)
    threshold = 0.6
    ds_syn = SyntheticDataset(synthetic_dir)
    ds_syn.transform = torch.nn.Identity()
    dataloader_syn = torch.utils.data.DataLoader(ds_syn, batch_size=bs,collate_fn=syn_collate_fn, shuffle=False, num_workers=4)
    # image = ds.get_image_by_idx(0)
    # filtered_array = []
    positive_confidence=[]

    for batch in tqdm.tqdm(dataloader_syn, total=len(dataloader_syn)):

        images = batch['pixel_values']
        inputs = processor(text=FILTER_CAPTIONS_MAPPING[args.dataset], images=images, return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        probs = probs.cpu().detach().numpy()

        # filtered_array = np.concatenate((filtered_array, np.where(probs[:,0] < threshold, 1, 0)))
        # filtered_count = np.sum(filtered_array).astype(int)
        # num = len(filtered_array)
        # filtered_ratio = np.round(filtered_count/num,4)
        # print(f"filtered_ration %{filtered_ratio*100}", f"{filtered_count}/{num}")
        positive_confidence = np.concatenate((positive_confidence, probs[:,0]))
    # filter the least 10% confident samples 
    positive_confidence = np.array(positive_confidence)
    bottom_threshold = np.percentile(positive_confidence, 5)
    up_threshold = np.percentile(positive_confidence, 95)
    print("threshold:",threshold)
    # filtered_positive_confidence = positive_confidence[]
    # meta_df = copy.deepcopy(ds_syn.meta_df)
    # meta_df.loc[:,'Path'] = ds_syn.meta_df['Path'].apply(lambda x: os.path.join(_dir,'data',x))
    meta_df = pd.read_csv(os.path.join(synthetic_dir,'meta.csv'))
    meta_df1 = meta_df[positive_confidence >= bottom_threshold]
    meta_df2 = meta_df[positive_confidence < bottom_threshold]
    meta_df3 = meta_df[positive_confidence >= up_threshold]

    meta_df1.to_csv(os.path.join(synthetic_dir,'meta_5-100per.csv'), index=False)
    meta_df2.to_csv(os.path.join(synthetic_dir,'meta_0-5per.csv'), index=False)
    meta_df3.to_csv(os.path.join(synthetic_dir,'meta_95-100per.csv'), index=False)

# python clip_data_filter.py --dataset 'cub' -g 2 --synthetic_type mixup_uniform -b 200
# python clip_data_filter.py --dataset 'aircraft' -g 3 --synthetic_type mixup_uniform -b 200
# python clip_data_filter.py --dataset 'dog' -g 3 --synthetic_type mixup_uniform -b 200
if __name__ == '__main__':
    parsers=argparse.ArgumentParser()
    parsers.add_argument('--dataset', type=str, default='cub')
    parsers.add_argument('-g','--gpu', type=str, default='0')
    parsers.add_argument('--synthetic_type', type=str, default='mixup0.7_5shot')
    parsers.add_argument('-b', '--batch_size', type=int, default=200)
    args = parsers.parse_args()
    main(args)
