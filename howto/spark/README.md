# Spark Examples with MLRun <br>

## 1. Spark Job with MLRun
Using MLRun to run Spark job.
The Spark job will run a describe function, which generates profile report<br>
from an Apache Spark DataFrame (Based on pandas_profiling).<br>

For each column the following statistics - if relevant for the column type - are presented:

**Essentials:** `type`, `unique values`, `missing values`,

**Quantile statistics:** `minimum value`, `Q1`, `median`, `Q3`, `maximum`, `range`, `interquartile range`.

**Descriptive statistics:** `mean`, `mode`, `standard deviation`, `sum`, `median absolute deviation`,<br> 
                            `coefficient of variation`, `kurtosis`, `skewness`.<br>
                        
**Most frequent values:** for categorical data 

## 2.Spark Job with MLRun
Run a Spark job which reads a csv file and logs the dataset to MLRun database.<br>

This basic example can use as a schema for more complex workloads using MLRun and Spark.

##  3.Spark Job with Spark Operator
Using spark operator for running spark job over k8s.<br>

The `spark-on-k8s-operator` allows Spark applications to be defined in a declarative manner and supports one-time Spark applications with `SparkApplication` and cron-scheduled applications with `ScheduledSparkApplication`. <br>

When sending a request with MLRun to Spark operator the request contains your full application configuration including the code and dependencies to run (packaged as a docker image or specified via URIs), the infrastructure parameters, (e.g. the memory, CPU, and storage volume specs to allocate to each Spark executor), and the Spark configuration.

Kubernetes takes this request and starts the Spark driver in a Kubernetes pod (a k8s abstraction, just a docker container in this case). The Spark driver can then directly talk back to the Kubernetes master to request executor pods, scaling them up and down at runtime according to the load if dynamic allocation is enabled. Kubernetes takes care of the bin-packing of the pods onto Kubernetes nodes (the physical VMs), and will dynamically scale the various node pools to meet the requirements.

When using Spark operator the resources will be allocated per task, means scale down to zero when the tesk is done.

