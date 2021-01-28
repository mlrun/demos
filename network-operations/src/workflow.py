from kfp import dsl
from mlrun import mount_v3io, mlconf
import os
from nuclio.triggers import V3IOStreamTrigger, CronTrigger

funcs = {}
projdir = os.getcwd()
projdir_path = f"/{os.environ['V3IO_USERNAME']}{projdir[len('/User'):]}"
labeled_stream_path = os.path.join(projdir_path, 'streaming', 'labeled_stream')
container = 'users'
full_path_projdir = os.path.join('/', container, os.environ["V3IO_USERNAME"], projdir[6:])

# Define a specific hub url?
# mlconf.hub_url = 'https://raw.githubusercontent.com/mlrun/functions/{tag}/{name}/function.yaml'
# mlconf.hub_url |= '/User/functions/{name}/function.yaml'

model_inference_stream = os.path.join(full_path_projdir, 'streaming', 'predictions')
labeled_stream = os.path.join(full_path_projdir, 'streaming', 'labeled_stream')

webapi_url = 'http://v3io-webapi:8081'
model_inference_url = f'{webapi_url}{model_inference_stream}'
labeled_stream_url = f'{webapi_url}{labeled_stream}'

def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        # Add V3IO Mount
        f.apply(mount_v3io())
        
        # Always pull images to keep updates
        f.spec.image_pull_policy = 'Always'
    
    # Define inference-stream related triggers
    functions['s2p'].add_trigger('labeled_stream', V3IOStreamTrigger(container=container,
                                                                     path=labeled_stream_path,
                                                                     seekTo='earliest',
                                                                     partitions=[0],
                                                                     consumerGroup='s2p',
                                                                     name='labeled_stream'))
    functions['generator'].add_trigger('cron', CronTrigger(interval='1m'))
    functions['labeled_stream'].add_trigger('cron', CronTrigger(interval='1m'))
    functions['create_feature_vector'].add_trigger('cron', CronTrigger(interval='1m'))
    functions['serving'].add_trigger('cron', CronTrigger(interval='1m'))
                
        
@dsl.pipeline(
    name='Network Operations Demo',
    description='Train a Failure Prediction LGBM Model over sensor data'
)
def kfpipeline(
        # aggregate
        df_artifact = os.path.join(projdir, 'data', 'metrics.pq'),
        metrics = ['cpu_utilization', 'throughput', 'packet_loss', 'latency'],
        metric_aggs = ['mean', 'sum', 'std', 'var', 'min', 'max', 'median'],
        suffix = 'daily',
        window = 10,

        # describe
        describe_table = 'netops',
        describe_sample = 0.3,
        label_column = 'is_error',
        class_labels = [1, 0],
        plot_hist = True,
    
        # Feature selection
        k = 5,
        min_votes = 3,
    
        # Train
        sample_size      = -1,        # -n for random sample of n obs, -1 for entire dataset, +n for n consecutive rows
        test_size        = 0.1,       # 10% set aside
        train_val_split  = 0.75,      # remainder split into train and val
    
        # Test
        predictions_col = 'predictions',
    
        # Deploy
        deploy_streaming = True,
        aggregate_fn_url = 'hub://aggregate',
        streaming_features_table = os.path.join(projdir, 'streaming', 'features'),
        streaming_predictions_table = os.path.join(projdir, 'streaming', 'predictions'),
    
        # Streaming
        streaming_metrics_table = os.path.join(projdir, 'streaming', 'metrics'),
        generator_metrics_configuration = os.path.join(projdir, 'src', 'metric_configurations.yaml'),
    
        # labeled stream creator
        streaming_labeled_table = labeled_stream,
        
        # Concept drift
        deploy_concept_drift = True,
        secs_to_generate = 10,
        concept_drift_models = ['ddm', 'eddm', 'pagehinkley'],
        output_tsdb = os.path.join(projdir, 'streaming', 'drift_tsdb'),
        input_stream = labeled_stream_url,
        output_stream = os.path.join(projdir, 'streaming', 'drift_stream'),
        streaming_parquet_table =  os.path.join(projdir, 'streaming', 'inference_pq'),
    
        # Virtual drift
        results_tsdb_container = 'users',
        results_tsdb_table = os.path.join(full_path_projdir[7:], 'streaming', 'drift_magnitude')
    ):
    
    # Run preprocessing on the data
    aggregate = funcs['aggregate'].as_step(name='aggregate',
                                                  params={'metrics': metrics,
                                                          'metric_aggs': metric_aggs,
                                                          'suffix': suffix,
                                                          'window': window},
                                                  inputs={'df_artifact': df_artifact},
                                                  outputs=['aggregate'],
                                                  handler='aggregate',
                                                  image='mlrun/ml-models')

    describe = funcs['describe'].as_step(name='describe-aggregation',
                                        handler="summarize",  
                                        params={"key": f"{describe_table}_aggregate", 
                                                "label_column": label_column, 
                                                'class_labels': class_labels,
                                                'plot_hist': plot_hist,
                                                'plot_dest': 'plots/aggregation',
                                                'sample': describe_sample},
                                        inputs={"table": aggregate.outputs['aggregate']},
                                        outputs=["summary", "scale_pos_weight"])
    
    feature_selection = funcs['feature_selection'].as_step(name='feature_selection',
                                                           handler='feature_selection',
                                                           params={'k': k,
                                                                   'min_votes': min_votes,
                                                                   'label_column': label_column},
                                                           inputs={'df_artifact': aggregate.outputs['aggregate']},
                                                           outputs=['feature_scores', 
                                                                    'max_scaled_scores_feature_scores'
                                                                    'selected_features_count', 
                                                                    'selected_features'],
                                                           image='mlrun/ml-models')
    
    describe = funcs['describe'].as_step(name='describe-feature-vector',
                                            handler="summarize",  
                                            params={"key": f'{describe_table}_features', 
                                                    "label_column": label_column, 
                                                    'class_labels': class_labels,
                                                    'plot_hist': plot_hist,
                                                    'plot_dest': 'plots/feature_vector'},
                                            inputs={"table": feature_selection.outputs['selected_features']},
                                            outputs=["summary", "scale_pos_weight"])
    
    train = funcs['train'].as_step(name='train',
                                   params={"sample"          : sample_size, 
                                           "label_column"    : label_column,
                                           "test_size"       : test_size,
                                           "train_val_split" : train_val_split},
                                   inputs={"dataset"         : feature_selection.outputs['selected_features']},
                                   hyperparams={'model_pkg_class': ["sklearn.ensemble.RandomForestClassifier", 
                                                                    "sklearn.linear_model.LogisticRegression",
                                                                    "sklearn.ensemble.AdaBoostClassifier"]},
                                   selector='max.accuracy',
                                   outputs=['model', 'test_set'],
                                   image='mlrun/ml-models')
    
    test = funcs['test'].as_step(name='test',
                                 handler='test_classifier',
                                 params={'label_column': label_column,
                                         'predictions_column': predictions_col},
                                 inputs={'models_path': train.outputs['model'],
                                         'test_set': train.outputs['test_set']},
                                 outputs=['test_set_preds'],
                                 image='mlrun/ml-models')

    
    with dsl.Condition(deploy_streaming == True):
        
        # deploy the model using nuclio functions
        deploy = funcs['serving'].deploy_step(env={'model_path': train.outputs['model'],
                                                   'FEATURES_TABLE': streaming_features_table,
                                                   'PREDICTIONS_TABLE': streaming_predictions_table,
                                                   'prediction_col': predictions_col}, 
                                              tag='v1')

        # test out new model server (via REST API calls)
        tester = funcs["model_server-tester"].as_step(name='model-tester',
                                                      params={'addr': deploy.outputs['endpoint'], 
                                                              'model': "predictor",
                                                              'label_column': label_column},
                                                      inputs={'table': train.outputs['test_set']},
                                                      outputs=['test_set_preds'])
    
        # Streaming demo functions
        preprocessor = funcs['create_feature_vector'].deploy_step(env={ 'aggregate_fn_url': aggregate_fn_url,
                                                                'METRICS_TABLE': streaming_metrics_table,
                                                                'FEATURES_TABLE': streaming_features_table,
                                                                'metrics': metrics,
                                                                'metric_aggs': metric_aggs,
                                                                'suffix': suffix,
                                                                'base_dataset': train.outputs['test_set'],
                                                                'label_col': label_column}).after(tester)

        labeled_stream_creator = funcs['labeled_stream'].deploy_step(env={'METRICS_TABLE': streaming_metrics_table,
                                                                                  'PREDICTIONS_TABLE': streaming_predictions_table,
                                                                                  'OUTPUT_STREAM': streaming_labeled_table,
                                                                                  'label_col': label_column,
                                                                                  'prediction_col': predictions_col}).after(tester)

        generator = funcs['generator'].deploy_step(env={'SAVE_TO': streaming_metrics_table,
                                                        'SECS_TO_GENERATE': secs_to_generate,
                                                        'METRICS_CONFIGURATION_FILEPATH': generator_metrics_configuration}).after(preprocessor)
        
        with dsl.Condition(deploy_concept_drift == True):

            concept_builder = funcs['concept_drift'].deploy_step(skip_deployed=True)

            concept_drift = funcs['concept_drift'].as_step(name='concept_drift_deployer',
                                                           params={'models': concept_drift_models,
                                                                   'label_col': label_column,
                                                                   'prediction_col': predictions_col,
                                                                   'output_tsdb': output_tsdb,
                                                                   'input_stream': f'{input_stream}@cds',
                                                                   'output_stream': output_stream},
                                                           inputs={'base_dataset': test.outputs['test_set_preds']},
                                                           artifact_path=mlconf.artifact_path,
                                                           image=concept_builder.outputs['image']).after(labeled_stream_creator)

            s2p = funcs['s2p'].deploy_step(env={'window': 10,
                                                'features': metrics,
                                                'save_to': streaming_parquet_table,
                                                'base_dataset': test.outputs['test_set_preds'],
                                                'results_tsdb_container': 'users',
                                                'results_tsdb_table': results_tsdb_table,
                                                'mount_path': '/users/orz',
                                                'mount_remote': '/User'}).after(labeled_stream_creator)
    
