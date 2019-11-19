# Real-time face recognition with re-enforced learning

This example is using OpenCV, and Nuclio demonstrating end to end solution for real time face recognition over video stream, 
it consists of 5 MLRun and Nuclio functions:

1. import images of known person into the cluster file system
2. training the model using hog algorithm  (form [Notebook](./notebooks/face_recognition.ipynb)
3. Automated deployment of Nuclio model serving function (form [Notebook](./notebooks/nuclio_face_prediction.ipynb) 
4. client that records video , streams the data into iguazio file system and triggers the serving function 
5. Labeling Unknown images using streamlit  (form [Notebook](./streamlit/label_prompt.py)
 
<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The Example also demonstrate an [automated pipeline](./notebooks/face_recognition.ipynb) using MLRun and KubeFlow pipelines 
The Example aslo includes using iguazio key value store and TSDB(Time Series Data Base) to store meta data and tracking data
of  on the identified person
## Notebooks & Code

* [All-in-one: Import, launch training, deploy serving](notebooks/face_recognition.ipynb) * 
* [Serving function development and testing](notebooks/nuclio_face_prediction.ipynb)
* [client for streaming data into iguazio](./client/VideoCapture.py)
* [labeling unknown images and re-enforce learning ](./streamlit/label_prompt.py)  
  
  


