![banner](./docs/hackathon-banner.jpg)

# Getting Started with the MLOps for Good Hackathon

 The Hackathon focuses on MLOps. It is important all submissions address the challenges of operationalizing the machine-learning project, and not just focus on the mode innovation. MLOps typically includes the following steps:

 1. [Business problem definition](#business-problem-definition)
 2. [Data ingestion and acquisition](#data-ingestion)
 3. [Training pipeline](#training)
 4. [Model serving](#serving)
 5. [Automated CI pipeline](#ci)

This guide will help guide you on get started.

<a id="setup"></a>

## 0. Setup

<details markdown="1">

Go to the [**Devpost hackathon page**](https://mlopsforgood.devpost.com/) and click the "Join hackathon" button. After you login/signup to the Devpost website you will see all existing projects and you will be able to start your own project.

### Collaboration, Resources + More to Help You Build

Whether or not you have started building your project, [**the Slack channel**](https://go.iguazio.com/mlopslive/joincommunity) is the perfect place to connect with other developers! 

Not only can you bounce ideas off each other and ask other developers for technical help, but you can also use the Slack Channel to ask the Iguazio team questions! 

#### Teaming up

Looking for a team? [**The Slack channel**](https://go.iguazio.com/mlopslive/joincommunity) can also help you connect with potential teammates. You can also use the challenge Participants Tab to search for other developers looking to team up, check out their skills and message them to see if they want to…

### MLRun

[**MLRun**](https://mlrun.org) is an open-source end-to-end MLOps framework that can significantly help you in this hackathon. We recommend you take the time to install and go through some of its basic examples.

To set-up your own cluster, check out the [**MLRun installation guide**](https://docs.mlrun.org/en/latest/install.html) for more details.

Resources:

* [**Quick-start guide**](https://docs.mlrun.org/en/latest/quick-start.html)
* [**Getting-started tutorial**](https://docs.mlrun.org/en/latest/tutorial/index.html)
* [**Converting Research Notebook to Operational Pipeline With MLRun**](https://docs.mlrun.org/en/latest/howto/convert-to-mlrun.html)
* [**MLRun end-to-end Demos**](https://github.com/mlrun/demos/tree/hackathon)

### Managed Environment

We have set up several pre-configured clusters on Azure cloud, if you would like access to such a cluster, send us an email to <hackathon@iguazio.com>.  Supply is limited with first-come-first-served, so hurry up if you require this cluster. Refer to [**the documentation**](https://www.iguazio.com/docs/latest-release/) for more information.

</details>

<a id="business-problem-definition"></a>
## 1. Business Problem Definition

<details markdown="1">

This is a crucial step when working on your project which can often impact its success. Don't skip this step. Take the necessary time to think about the problem you would like to address in your project.

Don't worry if you don't come up with ideas right away or if your ideas sound far-fetched at first. It's best to not limit yourself at the first stage, write down every idea that comes up.

After you feel you have enough ideas to consider, evaluate each idea. Some of the criteria you should evaluate:

1. Alignment with the MLOps for Good topics
2. Feasibility to complete within the Hackathon timeframe.
3. Availability of data.
4. Any existing models that address this problem.
5. Skillset and the number of people required to complete the project.

You may find that you have more than a single idea that may be worthwhile. Feel free to create more than one project or gage the interest of other participants. You may only be a member of a single project, but it's fun to see someone who manages to bring an idea you had to reality.

<a id="Data Ingestion and Acquisition"></a><a id="business-problem-definition"></a>
</details>


<a id="data-ingestion"></a>
## 2. Data Ingestion and Acquisition

<details markdown="1">

### Data Sources

Machine learning requires data, and therefore, you should research what data sources are available for your project. Consider the size of the data, if the dataset is too small, this may limit your ability to train your model. You should also think ahead about the serving process, and whether you can obtain new data for inference.

We recommend researching publicly available datasets, such as <https://github.com/awesomedata/awesome-public-datasets>. Before starting to use the dataset, please verify that the dataset owner gave the proper license/permission to use this dataset. If in doubt, contact the Iguazio team.

Additional open datasets:

- [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets.php)
- [Azure open datasets](https://azure.microsoft.com/en-us/services/open-datasets/catalog/)
- [Registry of open data on AWS](https://registry.opendata.aws/)
- [Google Cloud Platform Datasets](https://console.cloud.google.com/marketplace/browse?filter=solution-type:dataset)

While it is always best to build a solution based on real data, you may find the best way to implement your project is to generate data. In some cases the generated data is needed to enrich an existing dataset, in other cases, where data is scarce, you will need to generate the entire dataset. Data generation takes time and there are a lot of tools available for creating datasets. Do you research and try to find the tool that best fits your needs and skillset. Some tools may be generic, and others may be more specific. Make sure to include all the source code and documentation how to generate the files.

### Storage

If your data is not large and just a few files, it's likely the simplest approach is to read the files directly.

For are cases where you may find the need to store large amounts of data, it is best to consider an object storage, such as [**Azure Blob Storage**](https://azure.microsoft.com/en-us/services/storage/blobs/). In case you need you need to access data using different patterns, check out [**MongoDB Atlas**](https://www.mongodb.com/cloud/atlas).


If you choose to run your own cluster, but would like access to the multi-model in-memory data layer (V3IO), you can configure your assigned V3IO data layer credentials by specifying the `V3IO_API`, `V3IO_USERNAME` and `V3IO_ACCESS_KEY` environment variables. [send us an email](mailto:hackathon@iguazio.com>) and we will give you the credentials to this environment.

If you have other specific data access needs, define these requirements first, it is very likely you'll find an open-source project or a service that can be used for your project. 

</details>

<a id="training"></a>
## 3. Training pipeline

<details markdown="1">

### Feature Engineering

This is a common step when dealing with machine-learning project. Deep-learning projects sometimes do not require special feature engineering, for example, a neural-network can identify key areas in an image without special features.

In case you have to perform feature engineering, remember that the training pipeline is not just the model training. You may have some data manipulation to perform. Therefore, the data processing is still key to this step. You may not know ahead of time which features your model will need, this is an iterative process where you create a set of features, train some models, and then consider other features that may be useful. Consider not just the feature definition, but also how you would serve the features.

### Finding the Right Model

It's usually a good idea to try out several algorithms with different hyperparameters when training a model. While it's great if you come up with an innovative model, it's a good idea to look for previous work, maybe someone already has a relatively good model that you can use. Remember that this hackathon primary focus is not about the best model, but rather the best way to operationalize the workflow.

If you would like to take a more advanced approach, consider creating a model ensemble.

Resources to consider for model training:

* [**Azure automated machine learning**](https://azure.microsoft.com/en-us/services/machine-learning/automatedml/)
* [**MLRun serverless runtime**](https://docs.mlrun.org/en/latest/runtimes/functions.html)

</details>

<a id="serving"></a>
## 4. Model serving

<details markdown="1">

Even if you have built the most robust training pipeline, real world scenarios require running the model as part of an application. This step is where you would gain the most benefit from using an MLOps framework early on the start.

Model serving requires getting input data. In many applications this is an online source where the model provides some output (e.g., prediction) based on the stream. The model output is usually a response to the input and therefore it's also streaming. Other applications process files in batch, so you would need to consider where to put the input files, how to trigger the pipeline, and where to store the output.

You should also think about the interaction the user can have with the model. This is usually something simple, such as a basic web page or a dashboard (e.g., Grafana).

MLRun has [**serving and data pipeline**](https://docs.mlrun.org/en/latest/serving/index.html) capability which should make it easy to deploy the model. You can deploy your pipeline in a few lines of code.

</details>

<a id="ci"></a>
## 5. Automated CI pipeline

<details markdown="1">

You may want to go the extra mile and create an automated pipeline. For example, using GitHub actions to trigger model training when you provide new training file or deploy a model to serving once you train a new model or running a GitLab CI job.

See [**the MLRun documentation**](https://docs.mlrun.org/en/latest/ci-pipeline.html) for using MLRun in order to create a CI pipeline.

</details>

<a id="summary"></a>
## Summary

While it is tempting to start writing code right away, you will likely succeed by taking the time to research. Any existing publicly available data and models will significantly reduce the amount of effort. Also, consider what skillset is needed to complete your project and find the right teammates that can help you drive your project to completion.

We also recommend to be active in [**the Slack channel**](https://go.iguazio.com/mlopslive/joincommunity). Talking with other people can help discuss ideas together.
