# horovod
import horovod.tensorflow.keras as hvd

# mlrun 
from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact

# tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# sklean and plotting 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# generanl
import os
import numpy as np
import pandas as pd
import argparse
import glob
import cv2
import random as rand
import xml.etree.ElementTree as et
import re

# MLRun context
mlctx           = get_or_create_ctx('horovod-trainer')
imgs            = mlctx.get_param('imgs')
annot           = mlctx.get_param('annot')
model_artifacts = mlctx.get_param('model_artifacts')

# image batch and epocs
IMAGE_WIDTH     = mlctx.get_param('image_width', 224)
IMAGE_HEIGHT    = mlctx.get_param('image_height', 224)
IMAGE_CHANNELS  = mlctx.get_param('image_channels', 3)  # RGB color
IMAGE_SIZE      = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_SHAPE     = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
EPOCHS          = mlctx.get_param('epochs', 1)
BATCH_SIZE      = mlctx.get_param('batch_size', 16)

# parameters for reproducibility:
RANDOM_STATE    = mlctx.get_param('random_state', 1)
TEST_SIZE       = mlctx.get_param('test_size', 0.3)

# prep mapping
dic = {"image": [],"Dimensions": []}
for i in range(1,116):
    dic[f'Object {i}']=[]
    
print("Generating data in CSV format....")

for file in os.listdir(annot):
    row = []
    xml = et.parse(annot +"/" +file) 
    root = xml.getroot()
    img = root[1].text
    row.append(img)
    h,w = root[2][0].text,root[2][1].text
    row.append([h,w])

    for i in range(4,len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        row.append(temp)
    for i in range(len(row),119):
        row.append(0)
    for i,each in enumerate(dic):
        dic[each].append(row[i])
df = pd.DataFrame(dic)

# prep data
image_directories = sorted(glob.glob(os.path.join(imgs,"*.png")))

j=0
classes = ["without_mask","mask_weared_incorrect","with_mask"]
labels = []
data = []

for idx,image in enumerate(image_directories):
    img  = cv2.imread(image)
    #scale to dimension
    X,Y = df["Dimensions"][idx]
    cv2.resize(img,(int(X),int(Y)))
    #find the face in each object
    for obj in df.columns[3:]:
        info = df[obj][idx]
        if info!=0:
            label = info[0]
            info[0] = info[0].replace(str(label), str(classes.index(label)))
            info=[int(each) for each in info]
            face = img[info[2]:info[4],info[1]:info[3]]
            if((info[3]-info[1])>40 and (info[4]-info[2])>40):
                try:
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    data.append(face)
                    labels.append(label)
                    if(label=="mask_weared_incorrect"):
                        data.append(face)
                        labels.append(label)

                except:
                    pass
                
print("Done!")

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Horovod: initialize Horovod.
hvd.init()

# if gpus found, pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    device = 'GPU:0'
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    device = 'CPU:0'

print(f'Using device: {device}')

# Prepare, test, and train the data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=TEST_SIZE, stratify=labels, random_state=42)

# load model
baseModel = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False
    
# Horovod: adjust learning rate based on number of GPUs.
# opt = SGD(lr=0.001, momentum=0.9)
opt = Adam(lr=1e-4 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              experimental_run_tf_function=False,
              metrics=['accuracy'])

if hvd.rank() == 0:
    model.summary()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues, tensorflow.keras callback
    ReduceLROnPlateau(patience=10, verbose=1),
]

# for optimized gpu utilization please use tf.data
aug = ImageDataGenerator(
    zoom_range=0.1,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
    )

history = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=(len(trainX) // BATCH_SIZE) // hvd.size(),
    validation_data=(testX, testY),
    validation_steps=(len(testX) // BATCH_SIZE) // hvd.size(),
    epochs=EPOCHS)

# log artifact and results
if hvd.rank() == 0:
    # log the epoch advancement
    mlctx.logger.info('history:', history.history)
    print('MA:', model_artifacts)

    # Save the model file
    model.save('model.h5')
    # Produce training chart artifact
    chart = ChartArtifact('summary.html')
    chart.header = ['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss']
    for i in range(EPOCHS):
        chart.add_row([i + 1, history.history['accuracy'][i],
                       history.history['val_accuracy'][i],
                       history.history['loss'][i],
                       history.history['val_loss'][i]])
    summary = mlctx.log_artifact(chart, local_path='training-summary.html', 
                                 artifact_path=model_artifacts)


    # Save weights
    model.save_weights('model-weights.h5')
    weights = mlctx.log_artifact('model-weights', local_path='model-weights.h5', 
                                 artifact_path=model_artifacts, db_key=False)

    # Log results
    mlctx.log_result('loss', float(history.history['loss'][EPOCHS - 1]))
    mlctx.log_result('accuracy', float(history.history['accuracy'][EPOCHS - 1]))

    mlctx.log_model('model', artifact_path=model_artifacts, model_file='model.h5',
                    labels={'framework': 'tensorflow'},
                    metrics=mlctx.results, extra_data={
                        'training-summary': summary,
                        'model-architecture.json': bytes(model.to_json(), encoding='utf8')
                    })
