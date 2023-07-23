# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
