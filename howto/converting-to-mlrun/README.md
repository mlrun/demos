# How to convert an existing ML code to MLRun

This demo demonstrates how to convert an ML [script](https://www.kaggle.com/jsylas/python-version-of-top-ten-rank-r-22-m-2-88) into an end-to-end MLRun code. It ingest NYC taxi rides records, train a model that predicts the fare amount of a ride and deploy a serving function. 
The original ML script was copied into [original-code.ipynb](./original-code.ipynb) and converted into MLRun in [mlrun-code.ipynb](./mlrun-code.ipynb) and [model_serving_lightgbm.ipynb](./model_serving_lightgbm.ipynb).
