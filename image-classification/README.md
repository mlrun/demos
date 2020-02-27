# Image Classification Using Distributed Training

This demo uses TensorFlow, Keras, Horovod, and Nuclio to demonstrate an end-to-end solution for image recognition and classification.

The demo consists of four MLRun and Nuclio functions for perming the following tasks:

1. Import an image archive from AWS S3 to the data store of the Iguazio Data Science Platform ("the platform") &mdash; see the [**mlrun-mpijob-classify.ipynb**](mlrun-mpijob-classify.ipynb) notebook.
2. Tag the images based on their name structure &mdash; see the [**mlrun-mpijob-classify.ipynb**](mlrun-mpijob-classify.ipynb) notebook.
3. Perform distributed training using [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), and [Horovod](https://eng.uber.com/horovod/) &mdash; see the [**mlrun-mpijob-classify.ipynb**](mlrun-mpijob-classify.ipynb) notebook and the [**horovod_training.py**](horovod_training*.py) Python application.
4. Automate deployment of a [Nuclio](https://nuclio.io/) model serving function for serving the training model &mdash; from a web notebook &mdash; see [**nuclio-serving-tf-images.ipynb**](nuclio-serving-tf-images.ipynb) &mdash; or from a Dockerfile file &mdash; see the [**inference-docker**](inference-docker) directory, which contains a [**Dockerfile**](inference-docker/Dockerfile) and the related function code ([**main.py**](inference-docker/main.py)).

<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The demo also demonstrates ML pipeline automation from a web notebook using MLRun and [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) &mdash; see [**mlrun-mpijob-pipe.ipynb**](mlrun-mpijob-pipe.ipynb).

> **Note:** You can use the all-in-one [**mlrun-mpijob-classify.ipynb**](mlrun-mpijob-classify.ipynb) notebook to run all tasks.

