# Real-time face recognition with deep learning 

This example is using face recognition, OpenCV, scikit-learn and Nuclio demonstrating end to end solution for real time face recognition over video stream. 
<br>It consists of 5 MLRun and Nuclio functions:

<br>1. import images of known person into the cluster file system
<br>2. training the model using sklearn classifier on top of OpenCV deep learning algorithm (form [Notebook])(./notebooks/face_recognition.ipynb)
<br>3. Automated deployment of Nuclio model serving function (form [Notebook])(./notebooks/nuclio_face_prediction.ipynb) 
<br>4. client that records video , streams the data into iguazio file system and triggers the serving function 
<br>5. Labeling Unknown images using streamlit  (form [Code])(./streamlit/label_prompt.py)
 
<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The example also demonstrates an [automated pipeline](./notebooks/face_recognition.ipynb) using MLRun and KubeFlow pipelines, 
including using iguazio key value store and TSDB(Time Series Data Base) to store meta data and tracking data
of an the identified and un-identified faces.
## Notebooks & Code

* [All-in-one: Import, launch training, deploy serving](notebooks/face_recognition.ipynb) * 
* [Serving function development and testing](notebooks/nuclio_face_prediction.ipynb)
* [client for streaming data into iguazio](./client/VideoCapture.py)
* [labeling unknown images and re-enforce learning ](./streamlit/label_prompt.py)  
 
  


