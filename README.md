# MLRun Demos <!-- omit in toc -->

The mlrun/demos repository provides demos that implement full end-to-end ML use-case applications with MLRun and demonstrate different aspects of working with MLRun.

For more information about the MLRun Hackathon, refer to the [**hackathon getting-started**](./getting-started.md) section.

## In This Document <!-- omit in toc -->

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting-started Tutorial](#getting-started)
- [How-To: Converting Existing ML Code to an MLRun Project](#howto-convert-to-mlrun)
- [Integrating with CI Pipelines](#demo-ci-pipeline)
- [Model deployment Pipeline: Real-time operational Pipeline](#demo-model-deployment)
- [Healthcare Demo with Feature Store](#demo-healthcare)

<a id="overview"></a>
## Overview

The MLRun demos are end-to-end use-case applications that leverage [MLRun](https://github.com/mlrun/mlrun) to implement complete machine-learning (ML) pipelines &mdash; including data collection and preparation, model training, and deployment automation.

The demos demonstrate how you can

- Run ML pipelines locally from a web notebook such as Jupyter Notebook.
- Run some or all tasks on an elastic Kubernetes cluster by using serverless functions.

The demo applications are tested on the [Iguazio Data Science Platform](https://www.iguazio.com/) ("the platform") and use its shared data fabric, which is accessible via the `v3io` file-system mount; if you're not already a platform user, [request a free trial](https://www.iguazio.com/lp/14-day-free-trial-in-the-cloud/).

<a id="general-ml-workflow"></a>
### General ML Workflow

The provided demos implement some or all of the ML workflow steps illustrated in the following image:

<p><img src="./docs/mlrun-pipeline.png" alt="ML workflow" width="800"/></p>

<a id="prerequisites"></a>
## Prerequisites

To run the MLRun demos, first do the following:

- Prepare a Kubernetes cluster with preinstalled operators or custom resources (CRDs) for Horovod and/or Nuclio, depending on the demos that you wish to run.
- Install an MLRun service on your cluster.
  See the instructions in the [MLRun documentation](https://docs.mlrun.org/en/latest/install.html).
- Ensure that your cluster has a shared file or object storage for storing the data (artifacts).

<a id="getting-started"></a>
## Getting-started Tutorial

[**The tutorial**](./getting-started-tutorial/01-mlrun-basics.ipynb) covers MLRun fundamentals such as creation of projects and data ingestion and preparation, and demonstrates how to create an end-to-end machine-learning (ML) pipeline.
MLRun is integrated as a default (pre-deployed) shared service in the Iguazio Data Science Platform.

You'll learn how to

- Collect (ingest), prepare, and analyze data
- Train, deploy, and monitor an ML model
- Create and run an automated ML pipeline

You'll also learn about the basic concepts, components, and APIs that allow you to perform these tasks, including

- Setting up MLRun
- Creating and working with projects
- Creating, deploying and running MLRun functions
- Using MLRun to run functions, jobs, and full workflows
- Deploying a model to a serving layer using serverless functions


<a id="howto-convert-to-mlrun"></a>
## How-To: Converting Existing ML Code to an MLRun Project

The [**converting-to-mlrun**](./converting-to-mlrun/README.md) how-to demo demonstrates how to convert existing ML code to an MLRun project.
The demo implements an MLRun project for taxi ride-fare prediction based on a [Kaggle notebook](https://www.kaggle.com/jsylas/python-version-of-top-ten-rank-r-22-m-2-88) with an ML Python script that uses data from the [New York City Taxi Fare Prediction competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction).

The code includes the following components:

1. Data ingestion
2. Data cleaning and preparation
3. Model training
4. Model serving

<a id="converting-to-mlrun-pipeline-output"></a>
**Pipeline Output**

<p><img src="./docs/converting-to-mlrun-pipeline.png" alt="converting-to-mlrun pipeline output"/></p>

<a id="demo-ci-pipeline"></a>
## Integrating with CI Pipelines

The [**CI Pipeline**](./ci-pipeline/ci-pipeline.ipynb) demo demonstrates how to build a full end-to-end automated-ML pipeline using [scikit-learn](https://scikit-learn.org) and the UCI [Iris data set](http://archive.ics.uci.edu/ml/datasets/iris).

Users may want to run their ML Pipelines using CI frameworks like Github Actions, GitLab CI/CD, etc. MLRun support simple and native integration with the CI systems, see the following example in which we combine local code (from the repository) with MLRun marketplace functions to build an automated ML pipeline which:

- Runs data preparation
- Train a model
- Test the trained model
- Deploy the model into a cluster
- Test the deployed model

The demo by default uses Slack notifications and requires setting up webhooks. This process is straightforward and should take a few minutes. For more information see [the slack documentation](https://api.slack.com/messaging/webhooks)

<p><img src="./docs/trees.png" alt="scikit-learn tress image" width="500"/></p>

<a id="demo-model-deployment"></a>

## Model deployment Pipeline: Real-time operational Pipeline

This demo shows how to deploy a model with streaming information.

This demo is comprised of several steps:

<p><img src="./model-deployment-pipeline/assets/model-deployment-pipeline.png" alt="Model deployment Pipeline Real-time operational Pipeline" width="500"/></p>

> **Note**: this demo uses the Iguazio multi-model data layer (V3IO), primarily for real-time streaming. To run this demo outside
> an Iguazio cluster, you will need to get credentials to access a V3IO system.

While this demo covers the use case of 1<sup>st</sup>-day churn, it is easy to replace the data, related features and training model and reuse the same workflow for different business cases.

These steps are covered by the following pipeline:

- **1. Data generator** — Generates events for the training and serving and Create an enrichment table (lookup values). 
- **2. Event handler** - Receive data from the input. This is a common input stream for all the data. This way, one can easily replace the event source data (in this case we have a data generator) without affecting the rest of this flow. It also store all incoming data to parquet files.
- **3. Stream to features** - Enrich the stream using the enrichment table and Update aggregation features using the incoming event handler.
- **4. Optional model training steps -**
 - **4.1 Get Data Snapshot** - Takes a snapshot of the feature table for training.
  - **4.2 Describe the Dataset** - Runs common analysis on the datasets and produces plots suche as histogram, feature importance, corollation and more.
  - **4.3 Training** - Runing training with multiple classification models.
  - **4.4 Testing** - Testing the best performing model.
- **5. Serving** - Serve the model and process the data from the enriched stream and aggregation features.
- **6. Inference logger** - We use the same event handler function from above but only its capability to store incoming data to parquet files.

<a id="demo-healthcare"></a>
## Healthcare Demo with Feature Store

This demo shows the usage of MLRun and the feature store. The demo will showcase:
- [**Data ingestion & preparation**](./healthcare-feature-store/01-ingest-datasources.ipynb)
- [**Model training & testing**](./healthcare-feature-store/02-create-training-model.ipynb)
- [**Real-time data & model pipeline**](./healthcare-feature-store/03-deploy-serving-model.ipynb)

Healthcare facilities need to closely monitor their patients and identify early signs that can indicate that medical intervention is necessary. Time is a key factor, the earlier the medical teams can attend to an issue, the better the outcome. This means an effective system that can alert of issues in real-time can save lives.

In this demo we will learn how to **Ingest** different data sources to our **Feature Store**. Specifically, this patient data has been successfully used to treat hospitalized COVID-19 patients prior to their condition becoming severe or critical. To do this we will use a medical dataset which includes three types of data:  
- **Healthcare systems**: Batch updated dataset, containing different lab test results (Blood test results for ex.).
- **Patient Records**: Static dataset containing general patient details.
- **Real-time sensors**: Real-Time patient metric monitoring sensor.  

<img src="./docs/_static/images/feature_store_demo_diagram.png" width="600px" />

