# **Scikit-Learn and KubeFlow With MLRun**

<img src="./images/my-keras-pipeline.png" width="1200" align="center"/>

In this set of notebooks and functions we build a **[scikit-learn](https://scikit-learn.org/stable/)** **[Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators)** into a **[KubeFlow](https://www.kubeflow.org/)** **[pipeline](https://www.kubeflow.org/docs/pipelines/)**.  The sklearn `Pipeline` component is composed of 2 steps and is embodied within the Kubeflow training step. It performs feature engineering, scaling and fitting using components that are themselves parameters, making the Kubeflow pipeline appear more generic and thus more amenable to further automation.

So for example, this Kubeflow pipeline could have its input models determined by 3 other pipelines that search over a wider sample of model structures. 

And with MLRun, each step, of each pipeline, could have its own highly optimized/customized runtime with metrics and other data logged by the MLRun db and accessible through the Iguazia data fabric. 


