import torch
from torch.nn import Module
import torchvision


class MaskDetectionMobilenetV2(Module):
    def __init__(self):
        super(MaskDetectionMobilenetV2, self).__init__()

        # The model will be based on MobileNetV2:
        self.mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=True)

        # Construct the head of the model that will be placed on top of the the base model:
        self.mask_detection = torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=(7, 7)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1280, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=128, out_features=2),
            torch.nn.Softmax(dim=1),
        )

        # Loop over layers in MobilenetV2 and freeze them so they will not be updated during the first training process:
        for child in self.mobilenet_v2.children():
            for parameter in child.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        x = self.mobilenet_v2.features(x)
        x = self.mask_detection(x)
        return x
