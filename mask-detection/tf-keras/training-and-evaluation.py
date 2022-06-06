import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

import mlrun
import mlrun.frameworks.tf_keras as mlrun_tf_keras


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
            image = keras.preprocessing.image.load_img(
                image_file, target_size=(224, 224)
            )
            image = keras.preprocessing.image.img_to_array(image)
            image = keras.applications.mobilenet_v2.preprocess_input(image)
            images.append(image)
            labels.append(label)

    # Convert the images and labels to NumPy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    # Perform one-hot encoding on the labels:
    labels = LabelBinarizer().fit_transform(labels)
    labels = keras.utils.to_categorical(labels)

    # Check if its an evaluation, if so, use the entire data:
    if is_evaluation:
        return images, labels

    # Split the dataset into training and validation sets:
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42,
    )

    # Construct the training image generator for data augmentation:
    image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    return (
        image_data_generator.flow(x_train, y_train, batch_size=batch_size),
        (x_test, y_test),
    )


def _get_model() -> keras.Model:
    """
    Create the Mask Detection model based on MobileNetV2.

    :returns: The Mask Detection model.
    """
    # The model will be based on MobileNetV2:
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=keras.layers.Input(shape=(224, 224, 3)),
    )

    # Construct the head of the model that will be placed on top of the the base model:
    head_model = base_model.output
    head_model = keras.layers.AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = keras.layers.Flatten(name="flatten")(head_model)
    head_model = keras.layers.Dense(128, activation="relu")(head_model)
    head_model = keras.layers.Dropout(0.5)(head_model)
    head_model = keras.layers.Dense(2, activation="softmax")(head_model)

    # Place the head FC model on top of the base model (this will become the actual model we will train):
    model = keras.Model(
        name="mask_detector", inputs=base_model.input, outputs=head_model
    )

    # Loop over layers in the base model and freeze them so they will not be updated during the first training process:
    for layer in base_model.layers:
        layer.trainable = False

    return model


def train(
    context: mlrun.MLClientCtx,
    dataset_path: str,
    batch_size: int = 32,
    lr: float = 1e-4,
    epochs: int = 3,
):
    """
    The training handler. Create the Mask Detection model and run training using the given parameters. The training is
    orchestrated by MLRun.frameworks.tf_keras.

    :param context:      The MLRun Function's context.
    :param dataset_path: Dataset path to get the datasets from.
    :param batch_size:   Batch size to use for the datasets.
    :param lr:           The learning rate for the Adam optimizer.
    :param epochs:       The amount of epochs to train.
    """
    # Get the datasets:
    training_set, validation_set = _get_datasets(
        dataset_path=dataset_path, batch_size=batch_size
    )

    # Get the model:
    model = _get_model()

    # Apply MLRun's interface for tf.keras:
    mlrun_tf_keras.apply_mlrun(model=model, model_name="mask_detector", context=context)

    # Initialize the optimizer:
    optimizer = keras.optimizers.Adam(lr=lr)

    # Compile the model:
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"],
    )

    # Train the head of the network:
    model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs,
        callbacks=[keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)],
        steps_per_epoch=35,
    )


def evaluate(
    context: mlrun.MLClientCtx, model_path: str, dataset_path: str, batch_size: int,
):
    """
    The evaluation handler. Load the Mask Detection model and run an evaluation on the given parameters. The evaluation
    is orchestrated by MLRun.frameworks.tf_keras.

    :param context:      The MLRun Function's context.
    :param model_path:   Path to the model object to evaluate.
    :param dataset_path: Dataset path to get the evaluation set from.
    :param batch_size:   Batch size to use for the evaluation.
    """
    # Get the dataset:
    x, y = _get_datasets(
        dataset_path=dataset_path, batch_size=batch_size, is_evaluation=True
    )

    # Apply MLRun's interface for tf.keras and load the model:
    model_handler = mlrun_tf_keras.apply_mlrun(
        model_path=model_path,
        context=context,
    )

    # Evaluate:
    model_handler.model.evaluate(x=x, y=y, batch_size=batch_size)
