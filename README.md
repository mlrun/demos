# End to End MLRun Demos

The following examples demonstrate complete machine learning pipelines which include data collection, data preparation, 
model training and automated deployment. 

The demos include running locally on a notebook, running on an elastic Kubernetes cluster using serverless functions,
and pipeline automation using KubeFlow pipelines 

## [XGBoost Classification with Hyper Parameters (Iris dataset)](xgboost/train_xgboost_serverless.ipynb)

Demonstrate a popular machine learning use case (iris dataset) and how to run training in parallel with hyper-parameters.

The first step is injecting the iris dataset, followed by parallel XGBoost training, and automated model deployment


## [Image Classification Using Distributed Training (Horovod)](image_classification/README.md)

Demonstrate a use case of image classification using TensorFlow, Keras and Horovod.

The demo include 4 steps: download images repository, label the images, 
run a distributed job over MPI (Horovod), deploy a model serving Nuclio function.


## [Real-time face recognition with re-enforced learning]()

Demonstrate real-time face images capture and recognition, and location tracking of identities.

This comprehensive e demo include multiple components: live images capture utility, images identification and tracking, 
labeling app (using Streamlit) to tag unidentified faces, and model training
 

## [Predictive Network/Telemetry Monitoring]()

Demonstrate ingestion of telemetry data from simulator or live stream, feature exploration, 
data preparation, model training, and automated model deployment.

## [Running Serverless Spark](spark/mlrun_sparkk8s.ipynb)

Demonstrate how the same spark job can run locally and as a distributed MLRun job over Kubernetes.
The Spark function can be incorporated as a step in various data preparation and machine learning scenarios.
 