# Tensorflow-Keras v1 Classifier: Paysim Credit Fraud

In this collection of notebooks we are going to explore both credit fraud within the  context ```mlrun```: ingestion, reporting and inference.  

Using their 2 notebooks, credit_nuclio... as defintion of network.

    1. ingest--TSDB based on steps (hours).
    2. report--create function to agg report for given date range.
    3. inference--create function that will predict if transaction is
    fraudulent based on model.


## Report

Create one query that returns a table aggregate for query date range.

## ML Model

Labels are provided, data split into train-validation-test sets.
Use provided keras model binary classification.