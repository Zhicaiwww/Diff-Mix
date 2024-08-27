import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid


def visualize_images(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    nrow: int = 4,
    show=False,
    save=True,
    outpath=None,
):

    if isinstance(images[0], Image.Image):
        transform = transforms.ToTensor()
        images_ts = torch.stack([transform(image) for image in images])
    elif isinstance(images[0], torch.Tensor):
        images_ts = torch.stack(images)
    elif isinstance(images[0], np.ndarray):
        images_ts = torch.stack([torch.from_numpy(image) for image in images])
    # save images to a grid
    grid = make_grid(images_ts, nrow=nrow, normalize=True, scale_each=True)
    # set plt figure size to (4,16)

    if show:
        plt.figure(
            figsize=(4 * nrow, 4 * (len(images) // nrow + (len(images) % nrow > 0)))
        )
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
        # remove the axis
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    img = Image.fromarray(grid.astype(np.uint8))
    if save:
        assert outpath is not None
        if os.path.dirname(outpath) and not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
        img.save(f"{outpath}")
    return img
