import os
import sys
import random
os.environ["DISABLE_TELEMETRY"] = 'YES'
sys.path.append('../')

from utils import visualize_images
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from semantic_aug.few_shot_dataset import SyntheticDataset
from semantic_aug.augmentations.textual_inversion import load_embeddings
from utils import DATASET_NAME_MAPPING, AUGMENT, parse_synthetic_dir

if __name__ == '__main__':
    device = 'cuda:1'
    dataset = 'aircraft'
    csv_file = 'meta_90-100per.csv'
    csv_file = 'meta_0-10-per.csv'
    row = 5
    column = 2
    synthetic_type='mixup_uniform'
    synthetic_dir = parse_synthetic_dir(dataset, synthetic_type=synthetic_type)
    
    for csv_file in ['meta_0-10per.csv', 'meta_90-100per.csv']:
        ds = SyntheticDataset(synthetic_dir,csv_file=csv_file)
        num = row * column 
        indices = random.sample(range(len(ds)), num )

        image_list = []
        for index in indices:
            image_list.append(ds.get_image_by_idx(index).resize((224,224)))

        outpath=f'figures/cases_filtered/{dataset}/{synthetic_type}_{csv_file}.png'
        visualize_images(image_list,nrow=row,show=False,save=True,outpath=outpath)
