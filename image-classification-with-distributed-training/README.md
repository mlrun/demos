# Image Classification with Distributed Training Demo

[Overview](#overview)&nbsp;| [Running the Demo](#demo-run)&nbsp;| [Demo Flow](#demo-flow)&nbsp;| [Pipeline Output](#pipeline-output)

## Overview

This demo demonstrates an end-to-end image-classification solution using [TensorFlow](https://www.tensorflow.org/) (versions 1 or 2), [Keras](https://keras.io/), [Horovod](https://eng.uber.com/horovod/), and [Nuclio](https://nuclio.io/).

You'll learn how to take a deep-learning Python code and run it as a distributed process using Horovod, with minimal code changes and no DevOps.

When the model is ready, you'll create a Nuclio function that runs in the inference layer.
The function requests and retrieves images and sends back a reply according to the results of the image-classification model.

<a id="demo-run"></a>
## Running the Demo

To run the demo and generate an automated ML pipeline with source control, open the [**project.ipynb**](./project.ipynb) notebook and run the code cells according to the instructions in the notebook.

<a id="demo-flow"></a>
## Demo Flow

The demo consists of four MLRun and Nuclio functions and a Kubeflow Pipelines orchestration:

1. **Download** &mdash; import an image archive from AWS S3 to your cluster's data store.
2. **Label** &mdash; tag the images based on their name structure.
3. **Training** &mdash; perform distributed training using TensorFlow, Keras, and Horovod.
4. **Inference** &mdash; automate deployment of a Nuclio model-serving function.
5. **Pipelines** &mdash; create an automated pipeline.

> **Note:** The demo supports both TensorFlow versions 1 and 2.
> There's one shared notebook and two code files &mdash; one for each TensorFlow version.

The following image demonstrates the demo's workflow:

<p><img src="../docs/hvd-flow.png" alt="Demo workflow" width="600"/></p>

<a id="pipeline-output"></a>
## Pipeline Output

The following image illustrates the generated pipeline:

<p><img src="../docs/hvd-pipe.png" alt="Pipeline output" width="500"/></p>

