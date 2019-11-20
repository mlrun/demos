# Predictive Network/Telemetry Monitoring

## **WIP**

The netops demo demonstrates predictive infrastructure monitoring: the application builds, trains, and deploys a machine-learning model for analyzing and predicting failure in network devices as part of a network operations (NetOps) flow.

## Demo structure

The demo is comprised of five main parts:

### [Generator](nuclio-generator.ipynb):

Using our open source deployment generator (Which you can pip install here) we create a network deployment (Defaults to Company, Data center, Device).
We then add our metrics via metrics configuration. (Defaults to CPU Utilization, Latency, Throughput, Packet loss).

The generator can create both normal device metrics as defined by the Yaml, and error scenarios that cascade through the metrics until the device reaches a critical failure.

### [Data Preprocessing](nuclio-data-preperations.ipynb):

Turning the device's metrics stream to a feature vector using aggregations from multiple timespans (Current, Minutely, Hourly)

### [Training](nuclio-training.ipynb):

Using the feature vectors from the previous steps, and the is_error metrics given by the generator, train a ML model (Spans from scikit based to XGBoost & TF).
The model is then saved to a file for future usage.

### [Inference](nuclio-inference.ipynb):

Using the model file from the previous step and the feature vectors created by the Preprocessing stage, predict if a device is about to fail.

### [MLRun - Pipeline](mlrun.ipynb)

Using MLRun to create a pipeline from the four process nuclio functions.
