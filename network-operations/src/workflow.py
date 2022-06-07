import mlrun
from kfp import dsl


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="netops-demo")
def pipeline(
        vector_name,
        features=[],
        label_column="is_error",
        model_name="netops",
        model_pkg_class="sklearn.ensemble.RandomForestClassifier",
        start_time="now-7d",
        end_time="now",
):
    # Get FeatureVector
    get_vector = mlrun.run_function(
        "hub://get_offline_features",
        name="get_vector",
        params={'feature_vector': vector_name,
                'features': features,
                'label_feature': label_column,
                "start_time": start_time,
                "end_time": end_time,
                "entity_timestamp_column": "timestamp",
                'target': {'name': 'parquet', 'kind': 'parquet'},
                "update_stats": True},
        outputs=["feature_vector"],
    )

    auto_trainer = mlrun.import_function("hub://auto_trainer")

    # Train a models
    train = mlrun.run_function(
        auto_trainer,
        handler='train',
        params={"model_class": model_pkg_class},
        inputs={"dataset": get_vector.outputs['feature_vector']},
        outputs=["model", "test_set"],
    )

    # Test and visualize the model
    mlrun.run_function(
        auto_trainer,
        handler='evaluate',
        params={
            "label_columns": label_column,
            "model": train.outputs["model"]
        },
        inputs={"dataset": train.outputs["test_set"]},
    )

    # import the standard ML model serving function
    serving_fn = mlrun.import_function("hub://v2_model_server", new_name="serving")

    # set the serving topology to simple model routing
    # with data enrichment and imputing from the feature vector
    serving_fn.set_topology(
        "router",
        mlrun.serving.routers.EnrichmentModelRouter(
            feature_vector_uri=str(vector_name),
            impute_policy={"*": "$mean"}),
    )

    # Deploy the trained model as a serverless function
    mlrun.deploy_function(
        serving_fn, models=[{"key": model_name, "model_path": train.outputs["model"]}],
    )
