import os
from typing import Callable, List, Tuple

import mlrun
import mlrun.frameworks.pytorch as mlrun_torch
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode


class MaskDetectionDataset(Dataset):
    """
    The mask detection dataset, including the data augmentations and preprocessing.
    """

    def __init__(
        self, images: List[Image.Image], labels: Tensor, is_training: bool = True
    ):
        """
        Initialize a new dataset for training / evaluating the mask detection model.

        :param images:      The images.
        :param labels:      The labels.
        :param is_training: Whether to initialize a training set (apply the augmentations) or an evaluation / validation
                            set.
        """
        # Compose the transformations:
        augmentations = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(degrees=20),
                torchvision.transforms.RandomResizedCrop(
                    size=(224, 224),
                    ratio=(0.85, 1.15),
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )  # type: Callable[[Image.Image], Image.Image]
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )  # type: Callable[[Image.Image], Tensor]

        # Perform augmentations:
        if is_training:
            images = [augmentations(image) for image in images]

        # Preprocess the images:
        images = [preprocess(image) for image in images]

        # Store this dataset's data:
        self._images = images
        self._labels = labels.type(dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Return an image and its label at the given index.

        :param index: The index to get the dataset's item.

        :returns: The 'i' item as a tuple of tensors: (image, label).
        """
        return self._images[index], self._labels[index]

    def __len__(self) -> int:
        """
        Returns the amount of images in the dataset.

        :returns: The amount of images in the dataset.
        """
        return len(self._images)


class MaskDetector(Module):
    """
    The mask detector module, using MobileNetV2's features for transfer learning.
    """

    def __init__(self):
        """
        Initialize a model, downloading MobileNetV2's weights.
        """
        super(MaskDetector, self).__init__()

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

    def forward(self, x) -> Tensor:
        """
        Infer the given input through the model.

        :param x: An image to infer.

        :returns: The model's prediction.
        """
        x = self.mobilenet_v2.features(x)
        x = self.mask_detection(x)
        return x


def _get_datasets(
    dataset_path: str, batch_size: int, is_evaluation: bool = False,
):
    """
    Create the training and validation or evaluation datasets from the given path.

    :param dataset_path:  Path to the main directory with the with mask and without mask images directories.
    :param batch_size:    The batch size to use in the datasets.
    :param is_evaluation: Whether to return a tuple of training and evaluation datasets or just an evaluation dataset.

    :returns: If is_evaluation is False, a tuple of (Training dataset, Validation dataset). Otherwise, the Evaluation
              dataset.
    """
    # Build the dataset going through the classes directories and collecting the images:
    images = []
    labels = []
    for label, directory in enumerate(["with_mask", "without_mask"]):
        images_directory = os.path.join(dataset_path, directory)
        images_files = [
            os.path.join(images_directory, file)
            for file in os.listdir(images_directory)
            if os.path.isfile(os.path.join(images_directory, file))
        ]
        for image_file in images_files:
            images.append(
                Image.open(os.path.join(images_directory, image_file)).resize(
                    (224, 224)
                )
            )
            labels.append(label)

    # Perform one-hot encoding on the labels:
    labels = torch.tensor(labels)
    labels = torch.nn.functional.one_hot(labels)

    # Check if its an evaluation, if so, use the entire data:
    if is_evaluation:
        # Construct the dataset:
        evaluation_set = MaskDetectionDataset(images=images, labels=labels)
        # Construct the data loader:
        evaluation_set = DataLoader(
            dataset=evaluation_set, batch_size=batch_size, shuffle=False
        )
        return evaluation_set

    # Split the dataset into training and validation sets:
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42,
    )

    # Construct the datasets:
    training_set = MaskDetectionDataset(images=x_train, labels=y_train)
    validation_set = MaskDetectionDataset(
        images=x_test, labels=y_test, is_training=False
    )

    # Construct the data loaders:
    training_set = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    validation_set = DataLoader(
        dataset=validation_set, batch_size=batch_size, shuffle=False
    )

    return training_set, validation_set


def accuracy(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Accuracy metric.

    :param y_pred: The model's prediction.
    :param y_true: The ground truth.

    :returns: The accuracy metric value.
    """
    return 1 - (torch.norm(y_true - y_pred) / y_true.size()[0]).item()


def train(
    context: mlrun.MLClientCtx,
    dataset_path: str,
    batch_size: int = 32,
    lr: float = 1e-4,
    epochs: int = 3,
):
    """
    The training handler. Create the Mask Detection model and run training using the given parameters. The training is
    orchestrated by MLRun.frameworks.pytorch.

    :param context:      The MLRun Function's context.
    :param dataset_path: Dataset path to get the datasets from.
    :param batch_size:   Batch size to use for the datasets.
    :param lr:           The learning rate for the Adam optimizer.
    :param epochs:       The amount of epochs to train.
    """
    # Get the datasets:
    training_set, validation_set = _get_datasets(
        dataset_path=dataset_path, batch_size=batch_size,
    )

    # Initialize the model:
    model = MaskDetector()

    # Initialize the optimizer:
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    # Initialize the loss:
    loss = torch.nn.MSELoss()

    # Train the head of the network:
    mlrun_torch.train(
        model=model,
        training_set=training_set,
        loss_function=loss,
        optimizer=optimizer,
        validation_set=validation_set,
        metric_functions=[accuracy],
        epochs=epochs,
        training_iterations=35,
        model_name="mask_detector",
        custom_objects_map={"training-and-evaluation.py": "MaskDetector"},
        custom_objects_directory=os.path.join(os.path.dirname(dataset_path), "pytorch"),
        context=context,
    )


def evaluate(
    context: mlrun.MLClientCtx, model_path: str, dataset_path: str, batch_size: int,
):
    """
    The evaluation handler. Load the Mask Detection model and run an evaluation on the given parameters. The evaluation
    is orchestrated by MLRun.frameworks.pytorch.

    :param context:      The MLRun Function's context.
    :param model_path:   Path to the model object to evaluate.
    :param dataset_path: Dataset path to get the evaluation set from.
    :param batch_size:   Batch size to use for the evaluation.
    """
    # Get the dataset:
    evaluation_set = _get_datasets(
        dataset_path=dataset_path, batch_size=batch_size, is_evaluation=True
    )

    # Initialize the loss:
    loss = torch.nn.MSELoss()

    # Evaluate (the model will be loaded automatically from the provided model path):
    mlrun_torch.evaluate(
        model_path=model_path,
        dataset=evaluation_set,
        loss_function=loss,
        metric_functions=[accuracy],
        context=context,
    )
