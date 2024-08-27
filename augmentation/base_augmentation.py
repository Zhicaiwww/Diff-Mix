import abc
from typing import Any, Tuple

import torch.nn as nn
from PIL import Image


class GenerativeAugmentation(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(
        self, image: Image.Image, label: int, metadata: dict
    ) -> Tuple[Image.Image, int]:

        return NotImplemented


class GenerativeMixup(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(
        self, image: Image.Image, label: int, metadata: dict, strength: float
    ) -> Tuple[Image.Image, int]:

        return NotImplemented
