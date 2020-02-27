# LightGBM Classification with Hyperparameters (HIGGS Data Set)

This demo demonstrates how to implement a popular machine-learning competition use case &mdash; binary classification on the HIGGS data set &mdash; and run model training in parallel with hyperparameters.

The demo retrieves and stores the data in Parquet format, partitioning it into training, validation and test sets; runs parallel [LightGBM](https://github.com/microsoft/LightGBM) model training; and automates the model deployment.

To execute the demo, run the provided Jupyter notebooks in the following order and pay close attention to the instructions as you work through the cells.

> **Note:** 
> - Make sure to run all the cells, including the commented cells.
> - Don't try running an entire notebook in one go until you have worked through it, as there might be sections that should only be run once.

1. Set up the server using the [**model-server.ipynb**](model-server.ipynb) notebook.
2. Design a machine-learning pipeline using the [**kubeflow-pipeline.ipynb**](kubeflow-pipeline.ipynb) notebook.

Enjoy!
