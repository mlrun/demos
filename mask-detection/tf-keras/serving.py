import urllib.request
from typing import Dict, List, Union

import numpy as np
from PIL import Image
from tensorflow import keras


def resize(event: Dict) -> List[np.ndarray]:
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
        # Convert to numpy arrays:
        image = keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype="float32")
        # Collect it:
        resized_images.append(image)

    return resized_images


def preprocess(images: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """
    Run the given images through MobileNetV2 preprocessing so they will be ready to be inferred through the mask
    detection model.

    :param images: A list of images to preprocess.

    :returns: A dictionary for the TFKerasModelServer, with the preprocessed images in the 'inputs' key.
    """
    # Go through the given images and run MobileNetV2 preprocessing:
    preprocessed_images = [
        keras.applications.mobilenet_v2.preprocess_input(image) for image in images
    ]
    preprocessed_images = [np.vstack(preprocessed_images)]

    # Pack and return:
    return {"inputs": preprocessed_images}


def postprocess(model_response: dict) -> Dict[str, Union[int, float]]:
    """
    Read the predicted classes probabilities response from the TFKerasModelServer and parse them into a dictionary with
    the results.

    :param model_response: The TFKerasModelServer response with the predicted probabilities.

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
