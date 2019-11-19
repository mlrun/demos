# Image face recognition Using CV2 

This example is using CV2, and Nuclio demonstrating end to end solution for face recognition over video stream, 
it consists of 4 MLRun and Nuclio functions:

1. import images of known person into the cluster file system
2. client that records video , streams the data into iguazio file system and triggers the serving function 
4. training using pipelines
5. Automated deployment of Nuclio model serving function (form [Notebook](nuclio-serving-tf-images.ipynb) and from [Dockerfile](./inference-docker))

<br><p align="center"><img src="workflow.png" width="600"/></p><br>

The Example also demonstrate an [automated pipeline](mlrun_mpijob_pipe.ipynb) using MLRun and KubeFlow pipelines 

## Notebooks & Code

* [All-in-one: Import, launch training, deploy serving](face_recognition.ipynb) * 
* [Serving function development and testing](nuclio_face_prediction.ipynb)
* [Auto generation of KubeFlow pipelines workflow](mlrun_mpijob_pipe.ipynb)
* [client for streaming data into iguazio](./client)
  * [function code](./inference-docker/main.py)
  * [Dockerfile](./inference-docker/Dockerfile)


