import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = "/data/zhicai/datasets/waterbird_complete95_forest2water2/"


def onehot(size: int, target: int):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


class WaterBird(Dataset):
    def __init__(self, split=2, image_size=256, crop_size=224, return_onehot=False):
        self.root_dir = DATA_DIR
        dataframe = pd.read_csv(os.path.join(self.root_dir, "metadata.csv"))
        dataframe = dataframe[dataframe["split"] == split].reset_index()
        self.labels = list(
            map(lambda x: int(x.split(".")[0]) - 1, dataframe["img_filename"])
        )
        self.dataframe = dataframe
        self.image_paths = dataframe["img_filename"]
        self.groups = dataframe.apply(
            lambda row: f"{row['y']}{row['place']}", axis=1
        ).tolist()
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
        img = Image.open(os.path.join(self.root_dir, img_path)).convert("RGB")

        label = self.labels[idx]
        group = self.groups[idx]
        if self.transform:
            img = self.transform(img)
        if self.return_onehot:
            if isinstance(label, (int, np.int64)):
                label = onehot(self.num_classes, label)
        return img, label, group
