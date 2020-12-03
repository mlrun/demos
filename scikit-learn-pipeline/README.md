# scikit-learn Demo: Full AutoML Pipeline

[Overview](#overview)&nbsp;| [Running the Demo](#demo-run)&nbsp;| [Pipeline Output](#pipeline-output)

## Overview

This demo demonstrates how to build a full end-to-end automated-ML (AutoML) pipeline using [scikit-learn](https://scikit-learn.org) and the UCI [Iris data set](http://archive.ics.uci.edu/ml/datasets/iris).

The generated machine-learning (ML) pipeline, which also includes CI and data ingestion, consists of the following steps:

- Create an Iris data-set generator (ingestion) function.
- Ingest the Iris data set.
- Analyze the data-set features.
- Train and test the model using multiple algorithms (AutoML).
- Deploy the model as a real-time serverless function.
- Test the serverless function's REST API with a test data set.

See the [**sklearn-project.ipynb**](./sklearn-project.ipynb) notebook for details.

<p><img src="../docs/trees.png" alt="scikit-learn trees image" width="500"/></p>

<a id="demo-run"></a>
## Running the Demo

To run the demo, download the [**sklearn-project.ipynb**](./sklearn-project.ipynb) notebook into an empty directory and execute the code cells sequentially.

<a id="pipeline-output"></a>
## Pipeline Output

The output plots can be viewed as static HTML files in the [**plots**](./plots) directory.

<p><img src="../docs/skpipe.png" alt="pipeline output" width="500"/></p>

