import mlrun
from kfp import dsl

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="netops-demo")
def pipeline(
    vector_uri,
    label_column="is_error",
    model_name="netops",
    model_pkg_class="sklearn.ensemble.RandomForestClassifier",
):
    # Train a model
    train = mlrun.run_function(
        mlrun.import_function("hub://sklearn_classifier", new_name="train"),
        params={"label_column": label_column, "model_pkg_class": model_pkg_class},
        inputs={"dataset": vector_uri},
        outputs=["model", "test_set"],
    )

    # Test and visualize the model
    mlrun.run_function(
        mlrun.import_function("hub://test_classifier", new_name="test"),
        params={"label_column": label_column},
        inputs={
            "models_path": train.outputs["model"],
            "test_set": train.outputs["test_set"],
        },
    )

    # import the standard ML model serving function
    serving_fn = mlrun.import_function("hub://v2_model_server", new_name="serving")

    # set the serving topology to simple model routing
    # with data enrichment and imputing from the feature vector
    serving_fn.set_topology(
        "router",
        mlrun.serving.routers.EnrichmentModelRouter(
            feature_vector_uri=str(vector_uri),
            impute_policy={"*": "$mean"}),
    )

    # Deploy the trained model as a serverless function
    mlrun.deploy_function(
        serving_fn, models=[{"key": model_name, "model_path": train.outputs["model"]}],
    )
