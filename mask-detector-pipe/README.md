# Mask vs. No Mask Demo
## Image classification workflow with distributed training
The following example demonstrates an end to end data science workflow for building an an image classifier <br>
The model is trained on an images dataset of people with masks / no masks. <br>
Then the model is deployed as a function in a serving layer <br>
Users can send http request with an image of eople with masks / no masksimage and get a respond back that identify whether it has a mask or not.

This typical data science workflow comprises of the following:
* Download dataset
* Training a model on the images dataset
* Deploy a function with the new model in a serving layer
* Testing the function

Key technologies:
* Tensorflow-Keras for training the model
* Horovod for running a distributed training
* MLRun (open source library for tracking experiments https://github.com/mlrun/mlrun) for building the functions and tracking experiments
* Nuclio function for creating a funciton that runs the model in a serving layer

Based on: https://www.kaggle.com/notadithyabhat/face-mask-detector