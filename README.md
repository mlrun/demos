# End-to End MLRun Demos

<a id="overview"></a>
## Overview

The following examples demonstrate complete machine learning pipelines which include data collection, data preparation,
model training, validation and automated deployment.

The examples demonstrate how you can do the following:

- Run ML functions and pipelines locally on a notebook.
- Run some or all tasks on an elastic Kubernetes cluster using serverless functions.
- Create automated ML workflows using [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/).

The demo applications are tested on the [Iguazio's Data Science PaaS](https://www.iguazio.com/), and use Iguazio's shared data fabric (v3io), and can be modified to work with any shared file storage by replacing the `apply(v3io_mount())` calls with other Kubeflow volume modifiers.
You can request a [free trial of Iguazio PaaS](https://www.iguazio.com/lp/14-day-free-trial-in-the-cloud/).

### Examples

* [scikit-learn Pipeline with AutoML](#end-to-end-data-prep--scikit-learn-pipeline-with-automl-iris-data-set)
* [Image Classification Using Distributed Training (Horovod)](#image-classification-using-distributed-training-horovod)
* [Real-Time Face Recognition with Re-enforced Learning](#real-time-face-recognition-with-re-enforced-learning)
* [Predictive Network/Telemetry Monitoring](#predictive-networktelemetry-monitoring)
* [Real-time Customer Churn Prediction](#real-time-customer-churn-prediction-kaggle-telco-churn-dataset)

<a id="prerequisites"></a>
### Prerequisites

- A Kubernetes cluster with preinstalled operators/CRDs for Horovod, Nuclio, Spark (depending on the specific demo).
- MLRun Service installed in the cluster, [see instructions](https://github.com/mlrun/mlrun/blob/master/README.md#installation).
- Shared file or object storage for the data/artifacts.

<a id="general-ml-flow"></a>
### The General ML Pipeline Flow

The various demos follow some or all of the steps shown in the diagram below:

<br><p align="center"><img src="./docs/mlrun-pipeline.png" width="800"/></p><br>

<a id="demo-sklearn-pipe"></a>
## [End-to-End data prep + scikit-learn Pipeline with AutoML (Iris Data Set)](./sklearn-pipe/sklearn-project.ipynb)

Demonstrate a popular machine learning use case (iris data set), how to explore the data and build an end to end automated ML pipeline.

The combined CI/Data/ML pipeline includes the following steps:

- Build the iris generator (ingest) function container.
- Ingest the iris data.
- Analyze the data-set features.
- Train and test the model using multiple algorithms (AutoML).
- Deploy the model as a real-time serverless function.
- Test the serverless function REST API with a test data set.

To start, download the notebook [sklearn-project.ipynb](./sklearn-pipe/sklearn-project.ipynb) into an empty directory and run the cells one by one.

<br><p align="center"><img src="./docs/trees.png" width="500"/></p><br>

<a id="demo-sklearn-pipe-pipeline-output"></a>
#### Pipeline Output

You can see various output plots in [sklearn-pipe/plots](sklearn-pipe/plots) (static HTML files).

<br><p align="center"><img src="./docs/skpipe.png" width="500"/></p><br>

<a id="demo-horovd-image-classification"></a>
## [Image Classification Using Distributed Training (Horovod)](horovod-pipe/horovod-project.ipynb)

This example uses TensorFlow (v1 or v2), Horovod, and Nuclio, demonstrating end-to-end solution for image classification.
The demo consists of four MLRun and Nuclio functions and Kubeflow Pipelines orchestration:

1. **Download**: Import an image archive from S3 to the cluster file system.
2. **Label**: Tag the images based on their name structure.
3. **Training**: Distributed training using TF1 or TF2, Keras and Horovod.
4. **Inference**: Automated deployment of Nuclio model serving function.

> **Note:** The demo supports both TensorFlow v1 and v3, there is one (shared) notebook and two code files (one per TF version)

<br><p align="center"><img src="./docs/hvd-flow.png" width="600"/></p><br>

<a id="demo-horovd-image-classification-pipeline-output"></a>
#### Pipeline Output

<br><p align="center"><img src="./docs/hvd-pipe.png" width="500"/></p><br>

<a id="demo-face-recognition"></a>
## [Real-Time Face Recognition with Re-enforced Learning](faces/README.md)

Demonstrate real-time face image capture, recognition, and location tracking of identities.

This comprehensive demonstration includes multiple components: a live image capture utility, image identification and tracking, a labeling app to tag unidentified faces using Streamlit, and model training.

<a id="demo-face-recognition-pipeline-output"></a>
#### Pipeline Output

<br><p align="center"><img src="./faces/workflow.png" width="500"/></p><br>

<a id="demo-sklearn-pipe"></a>
## Predictive Network/Telemetry Monitoring
<!-- TODO: When the demo is read, edit the description, and remove the TBD. -->

**TBD under construction**

Demonstrate ingestion of telemetry data from simulator or live stream, feature exploration, data preparation (aggregation), model training, and automated model deployment.

The demo is maintained in a separate Git repository and also demonstrates how to manage project life cycle using git.

<br><p align="center"><img src="./docs/netops-metrics.png" width="500"/></p><br>

#### Pipeline Output

<br><p align="center"><img src="./docs/netops-pipe.png" width="500"/></p><br>

<a id="demo-churn"></a>
## [Real-time Customer Churn Prediction (Kaggle Telco Churn dataset)](./churn/README.md)

running customer churn data analyses using the **[Kaggle Telco Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)**, training and validating an XGBoost model, and serving that with real-time Nuclio functions.

The demo consists of few MLRun and Nuclio functions and Kubeflow Pipelines orchestration:
1. write custom data encoders:  raw data often needs to be processed, some features need to be categorized, other binarized.
2. summarize data: look at things like class balance, variable distributions.
3. define parameters and hyperparameters for a generic XGBoost training function
4. train and test a number of models
5. deploy a "best" models into "production" as a nuclio serverless functions
6. test the model servers

#### Pipeline Output

<br><p align="center"><img src="./churn/assets/pipeline-3.png" width="500"/></p><br>


