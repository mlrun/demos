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

* [All-in-one: Import, tag, launch training, deploy serving](mlrun_mpijob_classify.ipynb) 
* [Training function code](horovod-training.py)
* [Serving function development and testing](nuclio-serving-tf-images.ipynb)
* [Auto generation of KubeFlow pipelines workflow](mlrun_mpijob_pipe.ipynb)
* [Building serving function using Dockerfile](./inference-docker)
  * [function code](./inference-docker/main.py)
  * [Dockerfile](./inference-docker/Dockerfile)


