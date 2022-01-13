import os
import yaml
import pandas as pd
import datetime
import json
import mlrun
from typing import Dict
import copy
import random
import mlrun.feature_store as fstore

# Data generator
from v3io_generator import metrics_generator, deployment_generator


def _create_deployment(metrics_configuration, project=None):
    print("creating deployment")
    # Create meta-data factory
    dep_gen = deployment_generator.deployment_generator()
    faker = dep_gen.get_faker()

    # Design meta-data
    deployment_configs = metrics_configuration["deployment"]
    for level, level_configs in deployment_configs.items():
        dep_gen.add_level(
            level,
            number=level_configs["num_items"],
            level_type=getattr(faker, level_configs["faker"]),
        )

    # Create meta-data
    deployment_df = dep_gen.generate_deployment()

    # Add static metrics
    static_df = copy.copy(deployment_df)
    static_data_configs = metrics_configuration["static"]
    for static_feature, static_feature_values in static_data_configs.items():
        if str(static_feature_values).startswith("range"):
            static_df[static_feature] = [
                random.choice(eval(static_feature_values))
                for i in range(static_df.shape[0])
            ]
        else:
            static_df[static_feature] = [
                random.choice(static_feature_values) for i in range(static_df.shape[0])
            ]

    # Add stub data
    for metric, params in metrics_configuration["metrics"].items():
        value = params["distribution_params"].get("mu", 0)
        deployment_df[metric] = value

    if project:
        # save the simulated dataset for future use
        project.log_dataset("deployment", df=deployment_df, format="parquet")
        project.log_dataset("static", df=static_df, format="parquet")

    return deployment_df, static_df


def get_or_create_deployment(metrics_configuration, project=None, create_new=False):
    if project and not create_new:
        try:
            static_df = mlrun.get_dataitem(project.get_artifact_uri("static")).as_df()
            deployment_df = mlrun.get_dataitem(
                project.get_artifact_uri("deployment")
            ).as_df()
            return deployment_df.reset_index(), static_df
        except:
            pass

    # Create deployment
    return _create_deployment(metrics_configuration, project)


def get_data_from_sample(context, data: Dict, as_df: bool = False) -> Dict:
    deployment_levels = (
        context.deployment_levels
        if context and hasattr(context, "deployment_levels")
        else ["device"]
    )
    label_col_indicator = (
        context.label_col_indicator
        if context and hasattr(context, "label_col_indicator")
        else "is_error"
    )
    base_columns = deployment_levels + ["timestamp"]
    metrics = {k: v for k, v in data.items() if label_col_indicator not in k}
    labels = {
        k: v for k, v in data.items() if label_col_indicator in k or k in base_columns
    }

    if as_df:
        metrics = pd.DataFrame.from_dict(metrics)
        labels = pd.DataFrame.from_dict(labels)

    return metrics, labels


def get_sample(
    metrics_configuration: dict,
    as_df: bool = True,
    project=None,
    ticks=5,
    create_new=False,
):
    deployment_df, static_df = get_or_create_deployment(
        metrics_configuration, project=project, create_new=create_new,
    )
    initial_timestamp = int(
        os.getenv(
            "initial_timestamp",
            (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp(),
        )
    )
    met_gen = metrics_generator.Generator_df(
        metrics_configuration,
        user_hierarchy=deployment_df,
        initial_timestamp=initial_timestamp,
    )

    generator = met_gen.generate(as_df=True)
    for i in range(100):
        sample = next(generator)
    metrics_df, labels_df = get_data_from_sample(None, sample, as_df)
    for i in range(ticks):
        sample = next(generator)
        metrics2_df, labels2_df = get_data_from_sample(None, sample, as_df)
        metrics_df = metrics_df.append(metrics2_df)
        labels_df = labels_df.append(labels2_df)
    return metrics_df, labels_df, static_df


def init_context(context):

    # Get metrics configuration
    project = mlrun.get_run_db().get_project(mlrun.mlconf.default_project)
    params = project.params
    config = mlrun.get_dataitem(params["metrics_configuration_uri"]).get()
    metrics_configuration = yaml.safe_load(config)

    # Generate or create deployment
    deployment_df, static_deployment = get_or_create_deployment(
        metrics_configuration, project=project,
    )

    static_set = fstore.get_feature_set("static")
    fstore.ingest(static_set, static_deployment)

    setattr(context, "label_col_indicator", "error")
    setattr(context, "deployment_levels", ["device"])

    # Create metrics generator
    initial_timestamp = int(
        os.getenv(
            "initial_timestamp",
            (datetime.datetime.now() - datetime.timedelta(days=1)).timestamp(),
        )
    )
    met_gen = metrics_generator.Generator_df(
        metrics_configuration,
        user_hierarchy=deployment_df,
        initial_timestamp=initial_timestamp,
    )
    generator = met_gen.generate(as_df=True)
    setattr(context, "metrics_generator", generator)

    # Metrics pusher
    device_metrics_pusher = mlrun.datastore.get_stream_pusher(
        params["device_metrics_stream"]
    )
    setattr(context, "device_metrics_pusher", device_metrics_pusher)

    # Labels pusher
    device_labels_pusher = mlrun.datastore.get_stream_pusher(
        params["device_labels_stream"]
    )
    setattr(context, "device_labels_pusher", device_labels_pusher)


def handler(context, event):
    for i in range(10):
        # Generate sample from all devices in the network
        device_metrics = json.loads(
            next(context.metrics_generator)
            .reset_index()
            .to_json(orient="records", date_unit="s")
        )
        for metric in device_metrics:
            # Split the data to features and labels
            metrics, labels = get_data_from_sample(context, metric)

            # Push the data to the appropriate streams
            context.device_metrics_pusher.push(metrics)
            context.device_labels_pusher.push(labels)
    return metrics, labels
