# Faces Demo: Real-Time Image Recognition with Deep Learning

[Overview](#overview)&nbsp;| [Demo Flow](#demo-flow)&nbsp;| [Pipeline Output](#pipeline-output)&nbsp;| [Notebooks and Code](#notebooks-and-code)

## Overview

This demo demonstrates how to build a real-time face-recognition application.
It includes streaming video to the Iguazio Data Science Platform ("the platform"); running a face-recognition algorithm in real time; and building a face-tagging interface.
The application administrator can use the face-tagging interface to see the unrecognized faces and manually tag them, thus allowing the system to learn and train the model accordingly.
In addition, the demo implements location tracking of the identities determined by the faces recognition.

This demo uses [OpenCV](https://opencv.org/), [PyTorch](https://pytorch.org), [Streamlit](https://www.streamlit.io/), [Nuclio](https://nuclio.io/), and MLRun.
MLRun is used to build and track the functions.

<a id="demo-flow"></a>
## Demo Flow

The demo includes five MLRun and Nuclio functions for implementing the following flow:

1. **Face recognition and encoding in photos** using the OpenCV deep-learning model &mdash; from a [notebook](notebooks/face-recognition.ipynb).
2. **Model training and an output PyTorch predictor**, based on the extracted encodings &mdash; from a [notebook](notebooks/face-recognition.ipynb).
3. **Automated deployment of a model-serving Nuclio function** &mdash; from a [notebook](notebooks/nuclio-face-prediction.ipynb).
4. **A client** that records video, streams the data to the platform's data store (file system), and triggers the serving function (see [**client/README.md**](client/README.md)).
5. **Labeling of unrecognized faces using an interactive dashboard** that's built with Streamlit &mdash; from [code](./streamlit/dashboard.py) (see [**streamlit/README.md**](./streamlit/README.md)).

The demo also demonstrates how to build an automated pipeline from a [notebook](notebooks/face-recognition.ipynb) using MLRun and [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/), including using the platform's NoSQL (key-value) data store and time-series database (TSDB) library to store metadata and track data of the identified and unidentified faces.

<a id="pipeline-output"></a>
## Pipeline Output

The following image illustrates the generated pipeline:
 
<p><img src="workflow.png" alt="Pipeline output" width="600"/></p>

<a id="notebooks-and-code"></a>
## Notebooks and Code

- [**notebooks/face-recognition.ipynb**](notebooks/face-recognition.ipynb) &mdash; the main demo notebook ("all in one").
    Run this notebook to execute the entire pipeline &mdash; import, launch, train, and deploy a serving function..
- [**client/README.md**](client/README.md) &mdash; a video-streaming client.
- [**notebooks/nuclio-face-prediction.ipynb**](notebooks/nuclio-face-prediction.ipynb) &mdash; serving-function development and testing.
- [**client/video_capture.py**](client/video_capture.py) &mdash; a client for streaming data to the Iguazio Data Science Platform.
- [**streamlit/dashboard.py**](streamlit/dashboard.py) &mdash; labeling of unknown images, and model retraining for newly collected data.

## Running the demo 

1. open jupyter 
2. cd demos/faces
2. run face-recognition notebook &mdash; from a [notebook](notebooks/face-recognition.ipynb).
3. follow client readme to stream images into the system &mdash; from a [Readme](client/README.md). 
4. follow streamlit README to deploy strwamlit into the system &mdash; from a [Streamlit](streamlit/README.md).
