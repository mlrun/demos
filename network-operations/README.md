# NetOps Demo: Predictive Network Operations/Telemetry

[Overview](#overview)&nbsp;| [Running the Demo](#demo-run)&nbsp;| [Demo Flow](#demo-flow)&nbsp;|  [Notebooks and Code](#notebooks-and-code)

## Overview

This demo demonstrates how to build an automated machine-learning (ML) pipeline for predicting network outages based on network-device telemetry, also known as Network Operations (NetOps).
The demo implements both model training and inference, including model monitoring and concept-drift detection.
The demo simulates telemetry network data for running the pipeline.

The demo demonstrates how to

- Manage MLRun projects.
- Use GitHub as a source for functions to use in pipeline workflows.
- Use MLRun logging to track results and artifacts.
- Use MLRun's Feature Store for feature engineering.
- Deploy a live-endpoints production pipeline.

> **Note:** The demo applications are tested on the [Iguazio Data Science Platform](https://www.iguazio.com) ("the platform"), and use the platform's data store ("v3io").
> Contact [Iguazio support](mailto:support@iguazio.com) to request a free trial of the platform.

<a id="demo-run"></a>
## Running the Demo

<a id="demo-run-prerequisites"></a>
### Prerequisites

Before you begin, ensure that you have the following:

- A [Kubernetes](https://kubernetes.io/) and [Nuclio](https://nuclio.io/).
- An installation of MLRun with a running MLRun service and an MLRun dashboard.
    See details in the [MLRun README](https://github.com/mlrun/mlrun).

<a id="demo-execution-steps"></a>
### Execution Steps

Execute the following steps to run the demo:

1. Fork the [mlrun/demos](https://github.com/mlrun/demos) Git repository to your GitHub account.

2. In a client or notebook that is properly configured with MLRun, run the following code; replace `<your fork>` with the name of your mlrun/demos GitHub fork:
    ```
    mlrun project demos/network-operations/ -u git://github.com/<your fork>/demos/network-operations.git
    ```

3. Run the [**01-ingest.ipynb**](01-ingest.ipynb) notebook to create the feature sets and deploy the data generator and live ingestion endpoints.

4. Open the [**02-training-and-deployment.ipynb**](02-training-and-deployment.ipynb) notebook and follow the instructions to create a Feature Vector, and run an automated pipeline to train a model and deploy it for live real-time predictions.

<a id="demo-flow"></a>
## Demo Flow

The demo consists of:
1. Building and testing features from three sources (device metadata, real-time device metrics, and real-time device labels) using the feature store
2. Ingesting the data using batch (for testing) or real-time (for production)
3. Train and test the model with data from the feature-store
4. Deploying the model as part of a real-time feature engineering and inference pipeline
5. Real-time model and metrics monitoring, drift detection

<a id="feature-creation"></a>
### Feature Creation

In the feature creation stage we use MLRun's Feature Store to first define the features we are want to ingest and the operations we want to apply to them.  
In this demo we create 3 unique feature sets:
- **network-device metrics** &mdash; real-time network-device telemtry data such as cpu utilization, packet loss, latency, throughput. We will run real-time aggregations on top of these metrics with different time windows.
- **static network-device data** &mdash; static device data such as his model and manufacturing country. We will one-hot-encode the categorical features to have them ready for model ingestion.
- **network-device failure label indicator** &mdash; a stream representing the label responses. The Feature Store will later match them with the correct sample for dataset creation.


<a id="model-training"></a>
### Model Training

In the model training stage we will use MLRun's Feature Store to define the Feature Vector we want to use for our model from the features we created and the labels we ingest.  We will create a dataset from the feature vector and feed it to our [sklearn classifier training function](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb) from the [MLRun marketplace functions hub](https://github.com/mlrun/functions) and create a network-device failure prediction model.


<a id="model-Deployment-and-monitoring"></a>
### Model Deployment & Monitoring

In this stage we will use MLRun's model server to deploy our trained model to production.  We will use the `EnrichmentModelRouter` to retrieve the online feature vector automatically upon receiving a network-device id and predict device failures.  

We will then be able to use MLRun's Model Montoring through Grafana to see our model performance, feature analytics and drift metrics.


<a id="notebooks-and-code"></a>
## Notebooks and Code

<a id="notebooks"></a>
### Notebooks and Code

- [**01-ingest.ipynb**](01-ingest.ipynb) &mdash; the 1st demo step notebook. including project setup, genetaor deployment, feature sets creation and deployment.
- [**02-training-and-deployment.ipynb**](02-training-and-deployment.ipynb) &mdash; the 2nd demo step notebook. including feature vecto creation, dataset creation, model training, deployment and testing.
- [**src/generator.py**](src/generator.py) &mdash; a nuclio function to generate live network-device telemetry and publish it to a v3io stream.
- [**src/workflow.py**](src/workflow.py) &mdash; ML Pipeline for training, tests, and model deployment
<a id="project-cfg-files"></a>
### Project-Configuration Files

- [**src/metric_configurations.yaml**](src/metric_configurations.yaml) &mdash; a data-generator configurations file. defines the metrics for the demo's generator network-device telemetry data.

