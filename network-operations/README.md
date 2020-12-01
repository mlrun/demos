# Network-Operations (NetOps) Demo

This demo shows a full ML pipeline for error prediction based on network device telemetry using MLRun.

This demo demonstrates how to

- Manage MLRun projects.
- Use GitHub as a source for functions to use in pipeline workflows.
- Use MLRun logging to track results and artifacts.
- Run a Kubeflow pipeline using MLRun.
- Deploy a live-endpoints product pipeline.
- Deploy a concept-drift pipeline.

The demo applications are tested on the Iguazio's Data Science PaaS, and use Iguazio's shared data fabric (v3io), and can be modified to work with any shared file storage by replacing the `apply(v3io_mount())` calls with other Kubeflow volume modifiers (for example, `apply(mlrun.platforms.mount_pvc())`).
You can request a free trial of Iguazio PaaS.

<a id="demo-flow"></a>
## Demo Flow

This demo aims to show an example of a production system deployed fully by an automated pipeline.

The demo has three main parts:

- Training pipeline to create new model.
- Streaming pipeline endpoints for production.
- Detecting concept drift to monitor the model's performance.

You can select which parts of the workflow to run via the `deploy_streaming` flag for the Production deployment and the `deploy_concept_drift` for the drift detection flags.

For retraining purposes and scheduling of workflows you can use the [project runner]() function from the marketplace to create an HTTP endpoint (or any other nuclio trigger) that can run a workflow based on an event.

### Training Pipeline

The training pipeline includes the following process (based on the [MLRun functions marketplace](https://github.com/mlrun/functions)):

- [**Exploratory data analysis**](https://github.com/mlrun/functions/blob/master/describe/describe.ipynb) &mdash; provide histogram maps, class imbalance, correlation matrix, etc.
- [**Aggregation**](https://github.com/mlrun/functions/tree/master/aggregate/aggregate.ipynb) &mdash; run different rolling aggregations on top of the data set to create temporal features.
- [**Feature selection**](https://github.com/mlrun/functions/blob/master/feature_selection/feature_selection.ipynb) &mdash; select the best features to use by using a vote based on multiple metrics and basic model estimators.
- [**Training**](https://github.com/mlrun/functions/blob/master/sklearn_classifier/sklearn_classifier.ipynb) &mdash; train multiple scikit-learn (a.k.a. sklearn) API-based models (using automated hyperparameters search), and select the best model according to a selected metric.
- [**Test**](https://github.com/mlrun/functions/blob/master/test_classifier/test_classifier.ipynb) &mdash; use a dedicated test data set to provide performance benchmarks for the model.

### Production-Deployment Pipeline

In the production deployment phase, the aim is to deploy a full system to ingest new data, create the necessary features, and provide predictions.
The pipeline uses the Nuclio serverless runtime to create the live endpoints, which for ease-of-use as open source will work by passing Parquets files at a set directory between each other.
(Using a streaming engine would require many dependencies, which will make open-source deployment difficult.)

The demo's production process is composed of a set-up stage to start the [generator](notebooks/generator.ipynb), which mimics incoming data.
The pipeline has the following main components:

- [**Preprocessor**](notebooks/preprocessor.ipynb) &mdash; take the selected features and create them upon ingestion.
- [**Model Server**](notebooks/server.ipynb) &mdash; deploy the model to a live endpoint.
- [**Model-server tester**](https://github.com/mlrun/functions/blob/master/model_server_tester/model_server_tester.ipynb) &mdash; verify that the model endpoint is live and provides good predictions.
- [**Labeled-stream creator**](notebooks/labeled_stream_creator.ipynb) &mdash; join the incoming labels and predictions to a single source to assess the model's performance.

### Concept-Drift Deployment Pipeline

The concept-drift pipeline is made of two main components:

- [**Concept-drift detectors**](https://github.com/mlrun/functions/blob/master/concept_drift/concept_drift.ipynb) &mdash; use streaming drift detectors such as DDM and PH.
    Use a **"job"** to initialize the models with a base labeled data set and produce a live Nuclio endpoint to enlist to the labeled stream

- [**Drift magnitude**](https://github.com/mlrun/functions/blob/e236a6b006e9e5a095a93c4822e422ebce5ac2dc/virtual_drift/virtual_drift.ipynb) &mdash; take batches of data via Parquet, and apply multiple drift magnitude metrics such as TVD, Helinger, and KL Divergence to assess the drift between a base data set and the latest data.

<br><p align="center"><img src="./docs/run-pipeline.png"/></p><br>

<a id="demo-run"></a>
## Running the Demo

Before you begin, ensure that you have the following:

- A Kubernetes cluster with pre-installed Kubeflow, Nuclio.
- MLRun Service and UI installed, see the [MLRun README](https://github.com/mlrun/mlrun).

Then, execute the following steps to run the demo"

1. Clone this repo to your own Git.

2. In a client or notebook that is properly configured with MLRun and Kubeflow, run the following code; replace the `<...>` placeholder:
    ```
    mlrun project demos/network-operations/ -u git://github.com/<your-fork>/demos/network-operations.git
    ```

3. Run the [Generator](notebooks/generator.ipynb) notebook to create the metrics data set.

4. Open the [**project.ipynb**](project.ipynb) notebook and follow the instructions to develop and run an automated ML Pipeline.

> **Note:** Alternatively, you can run the `main` pipeline from the CLI and specify artifacts path using this code:
> ```sh
> mlrun project demos/network-operations/ -r main -p "/User/kfp/{{workflow.uid}}/"
> ```

## Notebooks and Code

- [Generator notebook (Generate metrics data set)](notebooks/generator.ipynb)
- [Preprocessor notebook](notebooks/preprocessor.ipynb)
- [Model server notebook](notebooks/server.ipynb)
- [Labeled stream creator](notebooks/labeled_stream_creator.ipynb)
- [Project creation and testing notebook](project.ipynb)

<a id="project-files"></a>
### Project Files

- [Project spec (functions, workflows, etc)](project.yaml)

<a id="workflow-code"></a>
### Workflow Code

- [Workflow code (init + dsl)](src/workflow.py)

<a id="pipeline-output"></a>
## Pipeline Output

<br><p align="center"><img src="./docs/workflow.png"/></p><br>

