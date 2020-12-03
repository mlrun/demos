# Network-Operations (NetOps) Demo

[Overview](#overview)&nbsp;| [Running the Demo](#demo-run)&nbsp;| [Demo Flow](#demo-flow)&nbsp;| [Pipeline Output](#pipeline-output)&nbsp;| [Notebooks and Code](#notebooks-and-code)

## Overview

This demo demonstrates how to build an automated machine-learning (ML) pipeline for predicting network outages based on network-device telemetry, also known as Network Operations (NetOps).
The demo implements both model training and inference, including model monitoring and concept-drift detection.
The demo simulates telemetry network data for running the pipeline.

The demo demonstrates how to

- Manage MLRun projects.
- Use GitHub as a source for functions to use in pipeline workflows.
- Use MLRun logging to track results and artifacts.
- Use MLRun to run a [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) pipeline.
- Deploy a live-endpoints production pipeline.
- Deploy a concept-drift pipeline.

> **Note:** The demo applications are tested on the [Iguazio Data Science Platform](https://www.iguazio.com) ("the platform"), and use the platform's data store ("v3io").
> However, they can be modified to work with any shared file storage by replacing the `apply(v3io_mount())` calls with other Kubeflow volume modifiers (for example, `apply(mlrun.platforms.mount_pvc())`).
> Contact [Iguazio support](mailto:support@iguazio.com) to request a free trial of the platform.

<a id="demo-run"></a>
## Running the Demo

<a id="demo-run-prerequisites"></a>
### Prerequisites

Before you begin, ensure that you have the following:

- A [Kubernetes](https://kubernetes.io/) cluster with installations of [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) and [Nuclio](https://nuclio.io/).
- An installation of MLRun with a running MLRun service and an MLRun dashboard.
    See details in the [MLRun README](https://github.com/mlrun/mlrun).

<a id="demo-execution-steps"></a>
### Execution Steps

Execute the following steps to run the demo:

1. Fork the [mlrun/demos](https://github.com/mlrun/demos) Git repository to your GitHub account.

2. In a client or notebook that is properly configured with MLRun and Kubeflow Pipelines, run the following code; replace `<your fork>` with the name of your mlrun/demos GitHub fork:
    ```
    mlrun project demos/network-operations/ -u git://github.com/<your fork>/demos/network-operations.git
    ```

3. Run the [**notebooks/generator.ipynb**](notebooks/generator.ipynb) notebook to create the metrics data set.

4. Open the [**project.ipynb**](project.ipynb) notebook and follow the instructions to develop and run an automated NetOps ML pipeline.

    > **Note:** Alternatively, you can use the following code to run the `main` pipeline from the CLI and specify the artifacts path:
    > ```sh
    > mlrun project demos/network-operations/ -r main -p "/User/kfp/{{workflow.uid}}/"
    > ```

<a id="demo-flow"></a>
## Demo Flow

The demo implements three main pipelines:

- [**Training pipeline**](#training-pipeline) &mdash; for training and creating new ML model.
- [**Production-deployment pipeline**](#production-deployment-pipeline) &mdash; for streaming pipeline endpoints to production.
    <br>
    To run this pipeline code, use the `deploy_streaming` flag.
- [**Concept-drift deployment pipeline**](#concep-drift-pipeline) &mdash; for monitoring the model's performance.
    <br>
    To run this pipeline code, use the `deploy_concept_drift` flag.

> **Note:** For model-retraining and workflow-scheduling purposes, you can use the [project runner](https://github.com/mlrun/functions/tree/master/project_runner) MLRun marketplace function to create an HTTP endpoint (or any other Nuclio trigger) that can run a workflow based on an event.

The following image demonstrates the demo workflow:

<p><img src="./docs/workflow.png" alt="Demo workflow"/></p>

<a id="training-pipeline"></a>
### Training Pipeline

The training pipeline includes the following elements, which use [MLRun marketplace functions](https://github.com/mlrun/functions):

- [**Exploratory data analysis**](https://github.com/mlrun/functions/blob/master/describe/describe.ipynb) &mdash; provide histogram maps, class imbalance, a correlation matrix, etc.
- [**Aggregation**](https://github.com/mlrun/functions/tree/master/aggregate/aggregate.ipynb) &mdash; run different rolling aggregations on top of the data set to create temporal features.
- [**Feature selection**](https://github.com/mlrun/functions/blob/master/feature_selection/feature_selection.ipynb) &mdash; select the best features to use by using a vote based on multiple metrics and basic model estimators.
- [**Training**](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb) &mdash; train multiple scikit-learn (a.k.a. sklearn) API-based models using automated hyperparameters search, and select the best model according to a selected metric.
- [**Test**](https://github.com/mlrun/functions/blob/master/test_classifier/test_classifier.ipynb) &mdash; use a dedicated test data set to provide performance benchmarks for the model.

<a id="production-deployment-pipeline"></a>
### Production-Deployment Pipeline

In the production-deployment phase, the aim is to deploy a full system for ingesting new data, creating the necessary features, and generating predictions.
The production-deployment pipeline uses the Nuclio serverless runtime to create the live endpoints.
For simplified open-source deployment, data is passed among the endpoints as Parquets files within a set directory.
(Using a streaming engine would require many dependencies, which complicates open-source deployment.)

The production pipeline is composed of a set-up stage to start the generator (see [**generator.ipynb**](notebooks/generator.ipynb)), which mimics incoming data.
The pipeline has the following main components:

- [**Preprocessor**](notebooks/preprocessor.ipynb) &mdash; creates the selected features upon ingestion.
- [**Model server**](notebooks/server.ipynb) &mdash; deploys the model to a live endpoint.
- [**Model-server tester**](https://github.com/mlrun/functions/blob/master/model_server_tester/model_server_tester.ipynb) &mdash; verifies that the model endpoint is live and provides good predictions.
- [**Labeled-stream creator**](notebooks/labeled_stream_creator.ipynb) &mdash; combines the incoming labels and predictions into a single source for assessing the model's performance.

<a id="concep-drift-pipeline"></a>
### Concept-Drift Pipeline

The concept-drift pipeline has two main components, which are based on the MLRun functions marketplace:

- [**Concept-drift detectors**](https://github.com/mlrun/functions/blob/master/concept_drift/concept_drift.ipynb) &mdash; streaming concept-drift detectors, such as DDM and PH.
    Use an MLRun job to initialize the models with a base labeled data set and produce a live Nuclio endpoint for the labeled stream.

- [**Drift magnitude**](https://github.com/mlrun/functions/blob/e236a6b006e9e5a095a93c4822e422ebce5ac2dc/virtual_drift/virtual_drift.ipynb) &mdash; applies multiple drift-magnitude metrics &mdash; such as TVD, Helinger, and KL Divergence &mdash; to Parquet data batches in order to asses the drift between a base data set and the latest data.

<a id="pipeline-output"></a>
## Pipeline Output

The following image illustrates the combined pipeline:

<p><img src="./docs/run-pipeline.png" alt="Pipeline output"/></p>

<a id="notebooks-and-code"></a>
## Notebooks and Code

<a id="notebooks"></a>
### Notebooks

- [**project.ipynb**](project.ipynb) &mdash; the main demo notebook ("all in one").
    Run this notebook to execute the entire pipeline.
- [**notebooks/generator.ipynb**](notebooks/generator.ipynb) &mdash; a metrics data-set generator.
- [**notebooks/preprocessor.ipynb**](notebooks/preprocessor.ipynb) &mdash; a preprocessor.
- [**notebooks/server.ipynb**](notebooks/server.ipynb) &mdash; a model server.
- [**notebooks/labeled_stream_creator.ipynb**](notebooks/labeled_stream_creator.ipynb) &mdash; a labeled-stream generator.

<a id="project-cfg-files"></a>
### Project-Configuration Files

- [**project.yaml**](project.yaml) &mdash; a project-configuration file, which defines the project's specification (functions, workflows, etc.).

<a id="workflow-code"></a>
### Workflow Code

- [**src/workflow.py**](src/workflow.py) &mdash; workflow code, including initialization and a definitive software library (DSL).

