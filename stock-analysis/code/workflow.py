from kfp import dsl
from mlrun import mount_v3io, mlconf, load_project
import os
from nuclio.triggers import V3IOStreamTrigger, CronTrigger
import re 

funcs = {}

# Directories and Paths
projdir = os.path.abspath('./')
project = load_project(projdir)
project_name = project.spec.params.get("PROJECT_NAME")
model_filepath = os.path.join(projdir, 'models', 'model.pt') # Previously saved model if downloaded
reviews_datafile = os.path.join(projdir, 'data', 'reviews.csv')
rnn_model_path = os.path.join(projdir, 'models', 'mymodel.h5')

# Performence limit
max_replicas = 1

# Readers cron interval
readers_cron_interval = '300s'

# Training GPU Allocation
# Set to 0 if no gpus are to be used
training_gpus = 0

def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        # Add V3IO Mount
        f.apply(mount_v3io())
        
        # Always pull images to keep updates
        f.spec.image_pull_policy = 'Always'
    
    # Define inference-stream related triggers
#     functions['sentiment_analysis_server'].add_model('bert_classifier_v1', model_filepath)
    functions['sentiment_analysis_server'].spec.readiness_timeout = 500
    functions['sentiment_analysis_server'].set_config('readinessTimeoutSeconds', 500)
    
    # Adept image to use CPU if a GPU is not assigned
    if training_gpus == 0:
        functions['sentiment_analysis_server'].spec.base_spec['spec']['build']['baseImage']='mlrun/ml-models'
        functions['bert_sentiment_classifier_trainer'].spec.image='mlrun/ml-models'
    
    # Add triggers
    functions['stocks_reader'].add_trigger('cron', CronTrigger(readers_cron_interval))
    functions['news_reader'].add_trigger('cron', CronTrigger(readers_cron_interval))
    
    
    # Set max replicas for resource limits
    functions['sentiment_analysis_server'].spec.max_replicas = max_replicas
    functions['news_reader'].spec.max_replicas = max_replicas
    functions['stocks_reader'].spec.max_replicas = max_replicas
    
    # Add GPU for training
    functions['bert_sentiment_classifier_trainer'].gpus(training_gpus)
        
@dsl.pipeline(
    name='Stocks demo deployer',
    description='Up to RT Stocks ingestion and analysis'
)
def kfpipeline(
    # General
    V3IO_CONTAINER = 'users',
    STOCKS_TSDB_TABLE = os.getenv('V3IO_USERNAME') + '/stocks/stocks_tsdb',
    STOCKS_KV_TABLE = os.getenv('V3IO_USERNAME') + '/stocks/stocks_kv',
    STOCKS_STREAM = os.getenv('V3IO_USERNAME') + '/stocks/stocks_stream',
    RUN_TRAINER: bool = False,
    
    # Trainer
    pretrained_model = 'bert-base-cased',
    reviews_dataset = reviews_datafile,
    models_dir = 'models',
    model_filename = 'bert_sentiment_analysis_model.pt',
    n_classes: int = 3,
    MAX_LEN: int = 128,
    BATCH_SIZE: int = 16,
    EPOCHS: int =  2,
    random_state: int = 42,
    
    # stocks reader
    STOCK_LIST: list = ['GOOGL', 'MSFT', 'AMZN', 'AAPL', 'INTC'],
    EXPRESSION_TEMPLATE = "symbol='{symbol}';price={price};volume={volume};last_updated='{last_updated}'",
    
    # Sentiment analysis server
    model_name = 'bert_classifier_v1',
    model_filepath = model_filepath # if not trained
    
    ):
    
    with dsl.Condition(RUN_TRAINER == True):
        
        deployer = funcs['bert_sentiment_classifier_trainer'].deploy_step()
                
        trainer = funcs['bert_sentiment_classifier_trainer'].as_step(name='bert_sentiment_classifier_trainer',
                                                                     handler='train_sentiment_analysis_model',
                                                                     params={'pretrained_model': pretrained_model,
                                                                             'EPOCHS': EPOCHS,
                                                                             'models_dir': models_dir,
                                                                             'model_filename': model_filename,
                                                                             'n_classes': n_classes,
                                                                             'MAX_LEN': MAX_LEN,
                                                                             'BATCH_SIZE': BATCH_SIZE,
                                                                             'EPOCHS': EPOCHS,
                                                                             'random_state': random_state},
                                                                     inputs={'reviews_dataset': reviews_dataset},
                                                                     outputs=['bert_sentiment_analysis_model'],
                                                                     image=deployer.outputs['image'])
        #becasue we switched to V2_Model_Server, no need to send model filepath as env variable
        sentiment_server = funcs['sentiment_analysis_server'].deploy_step()#env={f'SERVING_MODEL_{model_name}': trainer.outputs['bert_sentiment_analysis_model']}
        
        news_reader = funcs['news_reader'].deploy_step(env={'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_STREAM': STOCKS_STREAM,
                                                            'STOCKS_TSDB_TABLE': STOCKS_TSDB_TABLE,
                                                            'SENTIMENT_MODEL_ENDPOINT': sentiment_server.outputs['endpoint'],
                                                           'PROJECT_NAME' : project_name})
    
    with dsl.Condition(RUN_TRAINER == False):
        #becasue we switched to V2_Model_Server, no need to send model filepath as env variable
        sentiment_server = funcs['sentiment_analysis_server'].deploy_step() #env={f'SERVING_MODEL_{model_name}': model_filepath}
        
        news_reader = funcs['news_reader'].deploy_step(env={'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_STREAM': STOCKS_STREAM,
                                                            'STOCKS_TSDB_TABLE': STOCKS_TSDB_TABLE,
                                                            'SENTIMENT_MODEL_ENDPOINT': sentiment_server.outputs['endpoint'],
                                                           'PROJECT_NAME' : project_name})
    
    stocks_reader = funcs['stocks_reader'].deploy_step(env={'STOCK_LIST': STOCK_LIST,
                                                            'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_TSDB_TABLE': STOCKS_TSDB_TABLE,
                                                            'STOCKS_KV_TABLE': STOCKS_KV_TABLE,
                                                            'EXPRESSION_TEMPLATE': EXPRESSION_TEMPLATE,
                                                           'PROJECT_NAME' : project_name})
    
    stream_viewer = funcs['stream_viewer'].deploy_step(env={'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_STREAM': STOCKS_STREAM}).after(news_reader)
    
    vector_viewer = funcs['vector_reader'].deploy_step(env={'PROJECT_NAME' : project_name}).after(news_reader)
    
    
    rnn_model_training_deployer = funcs["rnn_model_training"].deploy_step(env={'model_path': rnn_model_path,
                                                                               'PROJECT_NAME' : project_name}).after(vector_viewer)
    
    rnn_serving = funcs['rnn_serving'].deploy_step().after(rnn_model_training_deployer)
    
    rnn_model_prediction = funcs["rnn_model_prediction"].deploy_step(env = {"endpoint":rnn_serving.outputs['endpoint']}).after(rnn_serving)
    
    
    grafana_viewer = funcs["grafana_view"].deploy_step()
    
    grafana_viewer = funcs["grafana_view"].as_step(params = {"streamview_url" : stream_viewer.outputs["endpoint"],
                                                             "readvector_url" : vector_viewer.outputs["endpoint"],
                                                             "rnn_serving_url" : rnn_model_prediction.outputs["endpoint"],
                                                             "v3io_container" : V3IO_CONTAINER,
                                                             "stocks_kv" : STOCKS_KV_TABLE,
                                                             "stocks_tsdb" : STOCKS_TSDB_TABLE,
                                                             "grafana_url" : "http://grafana"},
                                                   handler = "handler").after(grafana_viewer)
    
    
    
