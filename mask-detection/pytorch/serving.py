import urllib.request
from typing import Dict, List, Union

import numpy as np
import torchvision
from PIL import Image


def resize(event: Dict) -> List[Image.Image]:
    """
    Read images urls into numpy arrays and resize them to MobileNetV2 standard size of 224x224.

    :param event: A dictionary with the images urls at the 'data_url' key.

    :returns: A list of all the resized images as numpy arrays.
    """
    # Read the images urls passed:
    images_urls = event["data_url"]

    # Initialize an empty list for the resized images:
    resized_images = []

    # Go through the images urls and read and resize them:
    for image_url in images_urls:
        # Get the image:
        urllib.request.urlretrieve(image_url, "temp.png")
        image = Image.open("temp.png")
        # Resize it:
        image = image.resize((224, 224))
        # Collect it:
        resized_images.append(image)

    return resized_images


def preprocess(images: List[Image.Image]) -> Dict[str, List[np.ndarray]]:
    """
    Run the given images through MobileNetV2 preprocessing so they will be ready to be inferred through the mask
    detection model.

    :param images: A list of images to preprocess.

    :returns: A dictionary for the PyTorchModelServer, with the preprocessed images in the 'inputs' key.
    """
    # Prepare the transforms composition:
    transforms_composition = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Apply the transforms:
    preprocessed_images = [np.expand_dims(transforms_composition(image).numpy(), 0) for image in images]
    preprocessed_images = [np.vstack(preprocessed_images)]

    return {"inputs": preprocessed_images}


def postprocess(model_response: dict) -> Dict[str, Union[int, float]]:
    """
    Read the predicted classes probabilities response from the PyTorchModelServer and parse them into a dictionary with
    the results.

    :param model_response: The PyTorchModelServer response with the predicted probabilities.

    :returns: A dictionary with the parsed prediction.
    """
    # Read the prediction from the model:
    prediction = np.squeeze(model_response["outputs"])

    # Parse and return:
    return {
        "class": int(np.argmax(prediction)),
        "with_mask": float(prediction[0]),
        "without_mask": float(prediction[1]),
    }
