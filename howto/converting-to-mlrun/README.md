# How-To: Converting Existing ML Code to an MLRun Project

[Overview](#overview)&nbsp;| [Running the Demo](#demo-run)&nbsp;| [Demo Flow](#demo-flow)&nbsp;| [Pipeline Output](#pipeline-output)&nbsp;| [Notebooks and Code](#notebooks-and-code)

## Overview

This demo demonstrates how to convert existing machine-learning (ML) code to an MLRun project.
The demo implements an MLRun project for taxi ride-fare prediction based on a [Kaggle notebook](https://www.kaggle.com/jsylas/python-version-of-top-ten-rank-r-22-m-2-88) with an ML Python script that uses data from the [New York City Taxi Fare Prediction competition](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction).

<a id="demo-run"></a>
## Running the Demo

To run the demo, simply open the [**mlrun-code.ipynb**](./mlrun-code.ipynb) notebook from an environment with a running MLRun service and run the code cells.

<a id="demo-flow"></a>
## Demo Flow

The code includes the following components:

1. **Data ingestion** &mdash; ingest NYC taxi-rides data.
2. **Data cleaning and preparation** &mdash; process the data to prepare it for the model training.
3. **Model training** &mdash; train an ML model that predicts taxi-ride fares.
4. **Model serving** &mdash; deploy a function for serving the trained model.

<a id="pipeline-output"></a>
## Pipeline Output

<p><img src="../../docs/converting-to-mlrun-pipeline.png" alt="converting-to-mlrun pipeline output"/></p>

<a id="notebooks-and-code"></a>
## Notebooks and Code

- [**original-code.ipynb**](./original-code.ipynb) &mdash; contains the original ML script to be converted.
- [**mlrun-code.ipynb**](./mlrun-code.ipynb) &mdash; the main demo notebook ("all in one"), which contains the MLRun equivalent of the code from the **original-code.ipynb** notebook.
- [**model-serving.ipynb**](./model-serving.ipynb) &mdash; a model-serving function (used from **mlrun-code-.ipynb**).

