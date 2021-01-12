import os
from glob import glob

import numpy as np
import tensorflow as tf
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta

import horovod.tensorflow.keras as hvd

# Since we are not running within a specific function which
# receives `context` by MLRun as a parameter, we need to get
# the context explicitly.
mlctx = get_or_create_ctx("horovod-trainer")

# We can now use the context to get the parameters and inputs
# that were given to the function.
train_dir = mlctx.get_param("train_path")
val_dir = mlctx.get_param("val_path")
MODEL_DIR = mlctx.get_param("model_dir", "models")
CHECKPOINTS_DIR = mlctx.get_param("checkpoints_dir")
IMAGE_WIDTH = mlctx.get_param("image_width", 224)
IMAGE_HEIGHT = mlctx.get_param("image_height", 224)
IMAGE_CHANNELS = mlctx.get_param("image_channels", 3)  # RGB color
EPOCHS = mlctx.get_param("epochs", 1)
BATCH_SIZE = mlctx.get_param("batch_size", 16)
PREFETCH_STEPS = mlctx.get_param("prefetch_steps", 3)
LEARNING_RATE = mlctx.get_param("learning_rate", 1)
# RANDOM_STATE must be a parameter for reproducibility:
RANDOM_STATE = mlctx.get_param("random_state", 1)
TEST_SIZE = mlctx.get_param("test_size", 0.2)

IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

# Horovod: initialize Horovod.
hvd.init()

# if gpus found, pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    device = "GPU:0"
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
else:
    # If no GPUs are found, or we explicitely want to not use GPUs on the node,
    # set the device to CPU and verify via `cuda_visible_devices` that no GPU
    # will be used.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = "CPU:0"

print(f"Using device: {device}")

# We are going to use Transfer Learning from an imagenet
# trained model `ResNet50V2` for easy training.
model = tf.keras.applications.ResNet50V2(
    include_top=False, input_shape=IMAGE_SHAPE
)

# mark loaded layers as not trainable for transfer learning.
for layer in model.layers:
    layer.trainable = False

# add new classifier layers to learn our Cat/Dog specific classification
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
output = Dense(1, activation="sigmoid")(class1)

# define the final model
model = Model(inputs=model.inputs, outputs=output)

# Horovod: adjust learning rate based on number of GPUs.
opt = Adadelta(lr=LEARNING_RATE * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
# This Horovod wrapper is responsible for synchronizing the gradients
# between the Neural Networks running at each node between epochs.
opt = hvd.DistributedOptimizer(opt)

model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    experimental_run_tf_function=False,
    metrics=["accuracy"],
)

# Print the model from rank 0 only.
# Since the same code is running on all the nodes, we need to set a "master"
# rank for logging, saving, etc... (Or else this will be done by each node).
if hvd.rank() == 0:
    model.summary()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all
    # other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning
    # leads to worse final accuracy.
    # Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    # Reduce the learning rate if training plateaues, tensorflow.keras callback
    ReduceLROnPlateau(patience=10, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers
# from corrupting them.
if hvd.rank() == 0:
    callbacks.append(
        ModelCheckpoint(os.path.join(CHECKPOINTS_DIR, "checkpoint-{epoch}.h5"))
    )

# prep data for training using tf.data
train_files = tf.data.Dataset.list_files(str(train_dir), shuffle=False)
val_files = tf.data.Dataset.list_files(str(val_dir), shuffle=False)
class_names = np.array(sorted([dir1 for dir1 in os.listdir(train_dir)]))

train_num_files = len([file for file in glob(str(train_dir + "/*/*"))])
val_num_files = len([file for file in glob(str(val_dir + "/*/*"))])


# To process the label
def get_label(file_path):
    # convert the path to a list of path components separated by sep
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(tf.cast(one_hot, tf.int32))


# To process the image
def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])


# To create the single training of validation example with image and
# its corresponding label.
def process_TL(file_path):

    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    # Perform augmentations if needed
    # img= tf.image.rot90(img)
    # img=tf.image.flip_up_down(img)
    # img=tf.image.adjust_brightness(img,delta=0.1)
    img = preprocess_input(img)

    return img, label


# define the training and validation datasets
train_dataset = train_files.interleave(
    lambda x: tf.data.Dataset.list_files(
        str(train_dir + "/*/*"), shuffle=True
    ),
    cycle_length=4,
).map(process_TL, num_parallel_calls=tf.data.experimental.AUTOTUNE)

val_dataset = val_files.interleave(
    lambda x: tf.data.Dataset.list_files(str(val_dir + "/*/*"), shuffle=False),
    cycle_length=4,
).map(process_TL, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# for shuffling and a batch size 32 for batching
train_dataset = train_dataset.repeat().batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

# prefetch the data for better performence.
train_dataset = train_dataset.prefetch(buffer_size=PREFETCH_STEPS)
val_dataset = val_dataset.prefetch(buffer_size=PREFETCH_STEPS)

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=int((train_num_files) // BATCH_SIZE) // hvd.size(),
    callbacks=callbacks,
    epochs=EPOCHS,
    verbose=1 if hvd.rank() == 0 else 0,
    validation_data=val_dataset,
    validation_steps=int((val_num_files) // BATCH_SIZE) // hvd.size(),
)

# save the model only on worker 0 to prevent failures ("cannot lock file")
if hvd.rank() == 0:
    # os.makedirs(MODEL_DIR, exist_ok=True)
    model_artifacts = os.path.join(mlctx.artifact_path, MODEL_DIR)

    # log the epoch advancement
    mlctx.logger.info("history:", history.history)
    print("MA:", model_artifacts)

    # Save the model file
    model.save("model.h5")
    # Produce training chart artifact
    chart = ChartArtifact("summary.html")
    chart.header = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
    for i in range(EPOCHS):
        chart.add_row(
            [
                i + 1,
                history.history["accuracy"][i],
                history.history["val_accuracy"][i],
                history.history["loss"][i],
                history.history["val_loss"][i],
            ]
        )
    summary = mlctx.log_artifact(
        chart,
        local_path="training-summary.html",
        artifact_path=model_artifacts,
    )

    # Save weights
    model.save_weights("model-weights.h5")
    weights = mlctx.log_artifact(
        "model-weights",
        local_path="model-weights.h5",
        artifact_path=model_artifacts,
        db_key=False,
    )

    # Log results
    mlctx.log_result("loss", float(history.history["loss"][EPOCHS - 1]))
    mlctx.log_result(
        "accuracy", float(history.history["accuracy"][EPOCHS - 1])
    )
    mlctx.log_result(
        "val_accuracy", float(history.history["val_accuracy"][EPOCHS - 1])
    )
    mlctx.log_result(
        "val_loss", float(history.history["val_loss"][EPOCHS - 1])
    )

    # Log the model as a `model` artifact in MLRun.
    # You can log the different results and related files like
    # Relevant artifacts, plots, or data directly with the model artifact.
    # This way you can access all the relevant (and version-matched)
    # model artifacts directly with the model.
    mlctx.log_model(
        "model",
        artifact_path=model_artifacts,
        model_file="model.h5",
        labels={"framework": "tensorflow"},
        metrics=mlctx.results,
        extra_data={
            "training-summary": summary,
            "model-architecture.json": bytes(model.to_json(), encoding="utf8"),
            "model-weights.h5": weights,
        },
    )
