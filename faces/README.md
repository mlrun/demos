# Real-time face recognition with deep learning 

This example is using face recognition, OpenCV, PyTorch and Nuclio demonstrating end to end solution for real time face recognition over video stream. 
<br>It consists of 5 MLRun and Nuclio functions:

<br>1. face recognition and encoding in photos using OpenCV deep learning model. from [Notebook](notebooks/face-recognition.ipynb)
<br>2. train and ouput PyTorch predictor based on the extracted encodings. from [Notebook](notebooks/face-recognition.ipynb)
<br>3. automated deployment of Nuclio model serving function. from [Notebook](notebooks/nuclio-face-prediction.ipynb) 
<br>4. client that records video, streams the data into file system and triggers the serving function.  
<br>5. labeling unrecognized faces using interactive dashboard built with streamlit. from [Code](./streamlit/dashboard.py)
 
<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The example also demonstrates an [automated pipeline](notebooks/face-recognition.ipynb) using MLRun and KubeFlow pipelines, 
including using iguazio key value store and TSDB(Time Series Data Base) to store meta data and tracking data
of an the identified and un-identified faces.
## Notebooks & Code

* [Video Stream client](client/README.md)
* [All-in-one: import, launch training, deploy serving function](notebooks/face-recognition.ipynb)  
* [Serving function development and testing](notebooks/nuclio-face-prediction.ipynb)
* [client for streaming data into iguazio](client/video_capture.py)
* [labeling unknown images and retrain model on newly collected data](./streamlit/dashboard.py)  
