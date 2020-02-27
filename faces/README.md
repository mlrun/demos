# Real-time Face Recognition with Deep Learning 

This demo uses OpenCV, PyTorch, and Nuclio to demonstrate an end-to-end solution for real-time capture, recognition, and classification of face images over a video stream, as well as location tracking of identities.

The demo consists of five MLRun and Nuclio functions for performing the following tasks:

1. Identify (recognize) faces from webcam pictures and encode the processed images by using the [OpenCV](https://opencv.org/) deep-learning model &mdash; see the [**notebooks/face-recognition.ipynb**](notebooks/face-recognition.ipynb) notebook.

2. Train an model ML model and generate a predictor based on the extracted encodings by using [PyTorch](https://pytorch.org) &mdash; see the [**notebooks/face-recognition.ipynb**](notebooks/face-recognition.ipynb) notebook.

3. Automate deployment of a [Nuclio](https://nuclio.io/) function for serving the training model &mdash; see the [**notebooks/nuclio-face-prediction.ipynb**](notebooks/nuclio-face-prediction.ipynb) notebook.

4. Record video, stream the data to the data store of the Iguazio Data Science Platform ("the platform"), and trigger the serving function by using the demo's client application &mdash; see the [**client**](client/README.md) directory and files.

5. Classify and label unrecognized faces by using an interactive dashboard built with [Streamlit](https://www.streamlit.io/) &mdash; see the [**streamlit/dashboard.py**](streamlit/dashboard.py) Python application.
 
<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The demo also demonstrates ML pipeline automation from a web notebook using MLRun and [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/), including using the platform's NoSQL (key-value) data store and time-series database (TSDB) library to store metadata and track location data for the identified and unidentified faces &mdash; see [**notebooks/face-recognition.ipynb**](notebooks/face-recognition.ipynb).

