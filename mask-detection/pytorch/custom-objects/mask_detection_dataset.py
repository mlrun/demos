from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision

from PIL import Image


class MaskDetectionDataset(Dataset):
    def __init__(self, images: List[str], labels: Tensor):
        # Store this dataset's data:
        self._images = images
        self._labels = labels.type(dtype=torch.float32)

        # Compose the preprocessing transformations:
        self._preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomRotation(degrees=20),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # Prepare the image:
        image = Image.open(self._images[index])
        image = self._preprocess(image)

        # Prepare the label:
        label = self._labels[index]

        return image, label

    def __len__(self) -> int:
        return len(self._images)
