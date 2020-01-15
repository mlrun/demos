# LightGBM and KubeFlow with MLRun

<img src="./images/lgbm-pipeline.PNG" width="600" align="center"/>

In this set of notebooks we build a simple classification model using the **[HIGGS](https://archive.ics.uci.edu/ml/datasets/HIGGS)** dataset and the **[LightGBM](https://lightgbm.readthedocs.io/en/latest/)** package. The model is embedded into a **[KubeFlow](https://www.kubeflow.org/)** **[pipeline](https://www.kubeflow.org/docs/pipelines/)**.

Run the notebooks in the following order, and pay close attention to the instructions
as you work through the cells.  **Make sure to run all the cells, including the commented
cells**. Also, don't try running the entire workbook in one go until you have worked
through it, as there may be sections that should only be run once.

1. Setup the server using [model server](model-server.ipynb).
2. Design a machine learning pipeline using [kubeflow pipeline](kubeflow-pipeline.ipynb)

Enjoy!