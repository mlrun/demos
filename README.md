# MLRun Demos

The mlrun/demos repository provides full end-to-end ML demo use-case applications using MLRun.

#### In This Document

- [Overview](#overview)
  - [General ML Workflow](#general-ml-workflow)
- [Prerequisites](#prerequisites)
- [scikit-learn Demo: Full AutoML Pipeline](#demo-scikit-learn)
- [Image Classification with Distributed Training Demo](#demo-image-classification)
- [Faces Demo: Real-Time Image Recognition with Deep Learning](#demo-face-recognition)
- [Churn Demo: Real-Time Customer-Churn Prediction](#demo-churn)
- [NetOps Demo: Predictive Network Operations/Telemetry](#demo-netops)

<a id="overview"></a>
## Overview

The MLRun demos are end-to-end use-case applications that leverage [MLRun](https://github.com/mlrun/mlrun) to implement complete machine-learning (ML) pipelines &mdash; including data collection and preparation, model training, and deployment automation.

The demos demonstrate how you can

- Run ML pipelines locally from a web notebook such as Jupyter Notebook.
- Run some or all tasks on an elastic Kubernetes cluster by using serverless functions.
- Create automated ML workflows using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/).

The demo applications are tested on the [Iguazio Data Science Platform](https://www.iguazio.com/) ("the platform") and use its shared data fabric, which is accessible via the `v3io` file-system mount; if you're not already a platform user, [request a free trial](https://www.iguazio.com/lp/14-day-free-trial-in-the-cloud/).
You can also modify the code to work with any shared file storage by replacing the `apply(v3io_mount())` calls with any other Kubeflow volume modifier.

<a id="general-ml-workflow"></a>
### General ML Workflow

The provided demos implement some or all of the ML workflow steps illustrated in the following image:

<p><img src="./docs/mlrun-pipeline.png" alt="ML workflow" width="800"/></p>

<a id="prerequisites"></a>
## Prerequisites

To run the MLRun demos, first do the following:

- Prepare a Kubernetes cluster with preinstalled operators or custom resources (CRDs) for Horovod and/or Nuclio, depending on the demos that you wish to run.
- Install an MLRun service on your cluster.
  See the instructions in the [MLRun documentation](https://github.com/mlrun/mlrun/blob/master/README.md#installation).
- Ensure that your cluster has a shared file or object storage for storing the data (artifacts).

<a id="demo-scikit-learn"></a>
## scikit-learn Demo: Full AutoML Pipeline

The [**scikit-learn-pipeline**](./scikit-learn-pipeline/README.md) demo demonstrates how to build a full end-to-end automated-ML (AutoML) pipeline using [scikit-learn](https://scikit-learn.org) and the UCI [Iris data set](http://archive.ics.uci.edu/ml/datasets/iris).

The combined CI/data/ML pipeline includes the following steps:

- Create an Iris data-set generator (ingestion) function.
- Ingest the Iris data set.
- Analyze the data-set features.
- Train and test the model using multiple algorithms (AutoML).
- Deploy the model as a real-time serverless function.
- Test the serverless function's REST API with a test data set.

To run the demo, download the [**sklearn-project.ipynb**](./scikit-learn-pipeline/sklearn-project.ipynb) notebook into an empty directory and run the code cells according to the instructions in the notebook.

<p><img src="./docs/trees.png" alt="scikit-learn tress image" width="500"/></p>

<a id="demo-scikit-learn-pipeline-output"></a>
**Pipeline Output**

The output plots can be viewed as static HTML files in the [scikit-learn-pipeline/plots](scikit-learn-pipeline/plots) directory.

<p><img src="./docs/skpipe.png" alt="scikit-learn pipeline output" width="500"/></p>

<a id="demo-image-classification"></a>
## Image Classification with Distributed Training Demo

The [**image-classification-with-distributed-training**](image-classification-with-distributed-training/README.md) demo demonstrates an end-to-end image-classification solution using [TensorFlow](https://www.tensorflow.org/) (versions 1 or 2), [Keras](https://keras.io/), [Horovod](https://eng.uber.com/horovod/), and [Nuclio](https://nuclio.io/).

The demo consists of four MLRun and Nuclio functions and a Kubeflow Pipelines orchestration:

1. **Download** &mdash; import an image archive from AWS S3 to your cluster's data store.
2. **Label** &mdash; tag the images based on their name structure.
3. **Training** &mdash; perform distributed training using TensorFlow, Keras, and Horovod.
4. **Inference** &mdash; automate deployment of a Nuclio model-serving function.

> **Note:** The demo supports both TensorFlow versions 1 and 2.
> There's one shared notebook and two code files &mdash; one for each TensorFlow version.

<a id="demo-image-classification-demo-flow"></a>
**Demo Workflow**

<p><img src="./docs/hvd-flow.png" alt="Image-classification demo workflow" width="600"/></p>

<a id="demo-image-classification-pipeline-output"></a>
**Pipeline Output**

<p><img src="./docs/hvd-pipe.png" alt="Image-classification pipeline output" width="500"/></p>

<a id="demo-face-recognition"></a>
## Faces Demo: Real-Time Image Recognition with Deep Learning

The [**faces**](realtime-face-recognition/README.md) demo demonstrates real-time capture, recognition, and classification of face images over a video stream, as well as location tracking of identities.

This comprehensive demonstration includes multiple components:

- A live image-capture utility.
- Image identification and tracking using [OpenCV](https://opencv.org/).
- A labeling application for tagging unidentified faces using [Streamlit](https://www.streamlit.io/).
- Model training using [PyTorch](https://pytorch.org).
- Automated model deployment using [Nuclio](https://nuclio.io/)

<a id="demo-face-recognition-demo-flow"></a>
**Demo Workflow**

<p><img src="./realtime-face-recognition/workflow.png" alt="Face-recognition pipeline output" width="500"/></p>

<a id="demo-churn"></a>
## Churn Demo: Real-Time Customer-Churn Prediction

The [**chrun**](./customer-churn-prediction/README.md) demo demonstrates analysis of customer-churn data using the Kaggle [Telco Customer Churn data set](https://www.kaggle.com/blastchar/telco-customer-churn), model training and validation using [XGBoost](https://xgboost.readthedocs.io), and model serving using real-time Nuclio serverless functions.

The demo consists of few MLRun and Nuclio functions and a Kubeflow Pipelines orchestration:

1.  Write custom data encoders for processing raw data and categorizing or "binarizing" various features.
2.  Summarize the data, examining parameters such as class balance and variable distributions.
3.  Define parameters and hyperparameters for a generic XGBoost training function.
4.  Train and test several models using XGBoost.
5.  Identify the best model for your needs, and deploy it into production as a real-time Nuclio serverless function.
6.  Test the model server.

<a id="demo-churn-pipeline-output"></a>
**Pipeline Output**

<p><img src="./customer-churn-prediction/assets/pipeline-3.png" alt="Cutomer-churn pipeline output" width="500"/></p>

<a id="demo-netops"></a>
## NetOps Demo: Predictive Network Operations/Telemetry

The [NetOps demo](network-operations/README.md) demo demonstrates how to build an automated ML pipeline for predicting network outages based on network-device telemetry, also known as Network Operations (NetOps).
The demo implements both model training and inference, including model monitoring and concept-drift detection.
The demo simulates telemetry network data for running the pipeline.

The demo demonstrates how to

- Manage MLRun projects.
- Use GitHub as a source for functions to use in pipeline workflows.
- Use MLRun logging to track results and artifacts.
- Use MLRun to run a [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) pipeline.
- Deploy a live-endpoints production pipeline.
- Deploy a concept-drift pipeline.

The demo implements three pipelines:

- **Training pipeline** &mdash; ingestion of telemetry data, exploratory data analysis, data preparation (aggregation), feature selection, and model training and testing.
- **Production-deployment pipeline** &mdash; automated model deployment.
- **Concept-drift pipeline** &mdash; streaming of concept-drift detectors using a live Nuclio endpoint, and application of multiple drift-magnitude metrics to asses the drift between a base data set and the latest data.

<a id="demo-netops-pipeline-output"></a>
**Pipeline Output**

<p><img src="./docs/netops-pipe.png" alt="NetOps pipeline output" width="500"/></p>

