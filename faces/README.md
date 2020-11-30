# Faces Demo: Real-Time Image Recognition with Deep Learning

This demo uses [OpenCV](https://opencv.org/), [PyTorch](https://pytorch.org), and [Nuclio](https://nuclio.io/) to demonstrate an end-to-end solution for real-time face recognition over a video stream.

The demo consists 3 notebooks , client and images dashboard:

1. The main notebook - Face recognition will encode photos using the OpenCV , train the model based on deep-learning model and will deploy 2 nuclio functions video video-api-server and face prediction from other notebooks &mdash; from a [notebook](notebooks/face-recognition.ipynb).
2. Video api serving will handel the image sent by the client will use face predicition function to reply on the image request &mdash; from a [notebook](notebooks/nuclio-api-serving.ipynb).
3. Automated deployment of a Nuclio function for serving the training model &mdash; from a [notebook](notebooks/nuclio-face-prediction.ipynb).
4. A client that records video, streams the data to the platform's data store (file system), and triggers the serving function &mdash; see [**client/README.md**](client/README.md).
5. Labeling of unrecognized faces using an interactive dashboard built with [Streamlit](https://www.streamlit.io/) &mdash; from [code](./streamlit/dashboard.py).

The demo also demonstrates implementation of an automated pipeline from a [notebook](notebooks/face-recognition.ipynb) using MLRun and [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/), including using the Iguazio Data Science Platform's NoSQL (key-value) data store and time-series database (TSDB) library to store metadata and track data of the identified and unidentified faces.

<a id="pipeline-output"></a>
## Pipeline Output
 
<br><p align="center"><img src="workflow.png" width="600"/></p><br>

<a id="notebooks-and-code"></a>
## Notebooks and Code

- [Video-streaming client](client/README.md)
- [All-in-one: import, launch training, deploy a serving function](notebooks/face-recognition.ipynb)
- [Serving-function development and testing](notebooks/nuclio-face-prediction.ipynb)
- [Client for streaming data to the Iguazio Data Science Platform](client/video_capture.py)
- [Labeling of unknown images and model retraining for newly collected data](./streamlit/dashboard.py)

## Running the demo 


1. open jupyter 
2. cd demos/faces
2. run face-recognition notebook &mdash; from a [notebook](notebooks/face-recognition.ipynb).
3. follow client readme to stream images into the system &mdash; from a [Readme](client/README.md). 
4. follow streamlit README to deploy strwamlit into the system &mdash; from a [Streamlit](streamlit/README.md).
