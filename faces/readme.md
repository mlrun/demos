# Image face recognition Using CV2 

This example is using CV2, and Nuclio demonstrating end to end solution for real time face recognition over video stream, 
it consists of 4 MLRun and Nuclio functions:

1. import images of known person into the cluster file system
2. client that records video , streams the data into iguazio file system and triggers the serving function 
3. training the model using hog algorithm  (form [Notebook](./notebooks/nuclio_face_prediction.ipynb) 
4. Automated deployment of Nuclio model serving function (form [Notebook](./notebooks/nuclio_face_prediction.ipynb)
5. Labeling Unknown images using streamlit  (form [Notebook](./streamlit/label_prompt.py)
 
<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The Example also demonstrate an [automated pipeline](mlrun_mpijob_pipe.ipynb) using MLRun and KubeFlow pipelines 

## Notebooks & Code

* [All-in-one: Import, launch training, deploy serving](notebooks/face_recognition.ipynb) * 
* [Serving function development and testing](notebooks/nuclio_face_prediction.ipynb)
* [client for streaming data into iguazio](./client)
  * [running code](./client/VideoCapture.py)
  


