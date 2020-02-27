# Predictive Telemetry/Network Operations (NetOps) Monitoring

## **WORK IN PROGRESS**

This demo demonstrates predictive infrastructure monitoring: the application builds, trains, and deploys a machine-learning model for analyzing and predicting failure in network devices as part of a network-operations (NetOps) flow.

## Demo structure

The demo consists of five main parts:

### [Generator](nuclio-generator.ipynb)

Using the open-source deployment generator (which you can `pip install` here), create a network deployment (defaults to `Company`, `Data center`, and `Device`).
Then, add your metrics via metrics configuration (defaults to `CPU Utilization`, `Latency`, `Throughput`, and `Packet loss`).

The generator can create both normal device metrics, as defined in the YAML file, and error scenarios that cascade through the metrics until the device reaches a critical failure.

### [Data preprocessing](nuclio-data-preperations.ipynb)

Turning the device's metrics stream to a feature vector using aggregations from multiple time spans (current, ,minutely, or hourly)

### [Training](nuclio-training.ipynb)

Using the feature vectors from the previous steps, and the `is_error` metrics given by the generator, train an ML model; (spans from scikit-learn based to XGBoost and TensorFlow).
The model is then saved to a file for future usage.

### [Inference](nuclio-inference.ipynb)

Using the model file from the previous step and the feature vectors created in the preprocessing stage, predict whether a device is about to fail.

### [MLRun pipeline](mlrun.ipynb)

Using MLRun to create a pipeline from the four process Nuclio functions.

