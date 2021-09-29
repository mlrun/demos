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
artifact_path = project.spec.params.get("ARTIFACT_PATH")
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
    functions['sentiment_analysis_server'].spec.readiness_timeout = 500
    functions['sentiment_analysis_server'].set_config('readinessTimeoutSeconds', 500)
    
    # Adept image to use CPU if a GPU is not assigned
    if training_gpus == 0:
        functions['sentiment_analysis_server'].spec.base_spec['spec']['build']['baseImage']='mlrun/ml-models'
        functions['bert_sentiment_classifier_trainer'].spec.image='mlrun/ml-models'
    
    # Add triggers
    functions['stocks_reader'].add_trigger('cron', CronTrigger(readers_cron_interval))
    functions['news_reader'].add_trigger('cron', CronTrigger(readers_cron_interval))
    functions['rnn_model_training'].add_trigger('cron', CronTrigger('12h'))
    
    # Set max replicas for resource limits
    functions['sentiment_analysis_server'].spec.max_replicas = max_replicas
    functions['news_reader'].spec.max_replicas = max_replicas
    functions['stocks_reader'].spec.max_replicas = max_replicas
    
    # Add GPU for training
    functions['bert_sentiment_classifier_trainer'].gpus(training_gpus)
    
    project.func('news_reader').spec.max_replicas = 1

    # Declare function base image to build (Job and not a nuclio funciton)
    functions['func_invoke'].spec.image = "mlrun/mlrun"
    functions['bert_sentiment_classifier_trainer'].spec.build.commands = ['pip install transformers==3.0.1', 'pip install torch==1.6.0']
    functions['stocks_reader'].spec.build.commands = ['pip install lxml', 'pip install yfinance','pip install v3io_frames']
    functions['news_reader'].spec.build.commands = ['pip install beautifulsoup4', 'pip install v3io_frames']
    functions['stream_viewer'].spec.build.commands = ['pip install v3io']
    functions['grafana_view'].spec.build.commands = ['pip install git+https://github.com/v3io/grafwiz --upgrade', 'pip install v3io_frames', 'pip install attrs==19.1.0']
    functions['sentiment_analysis_server'].add_model("model1", class_name='SentimentClassifierServing', model_path=model_filepath)
    functions['rnn_serving'].add_model("model2",class_name="RNN_Model_Serving",model_path = rnn_model_path)  
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
                                                                     image=deployer.outputs['image']).after(deployer)
        
        sentiment_server = funcs['sentiment_analysis_server'].deploy_step().after(trainer)
        
        news_reader = funcs['news_reader'].deploy_step(env={'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_STREAM': STOCKS_STREAM,
                                                            'STOCKS_TSDB_TABLE': STOCKS_TSDB_TABLE,
                                                            'SENTIMENT_MODEL_ENDPOINT': sentiment_server.outputs['endpoint'],
                                                            'PROJECT_NAME' : project_name,
                                                            'ARTIFACT_PATH' : artifact_path}).after(sentiment_server)
        
        news_reader_invok1 = funcs['func_invoke'].as_step(params = {"endpoint" : news_reader.outputs["endpoint"]},
                                                             handler="handler").after(news_reader)
            
    with dsl.Condition(RUN_TRAINER == False):
        sentiment_server = funcs['sentiment_analysis_server'].deploy_step()
        
        news_reader = funcs['news_reader'].deploy_step(env={'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_STREAM': STOCKS_STREAM,
                                                            'STOCKS_TSDB_TABLE': STOCKS_TSDB_TABLE,
                                                            'SENTIMENT_MODEL_ENDPOINT': sentiment_server.outputs['endpoint'],
                                                            'PROJECT_NAME' : project_name,
                                                            'ARTIFACT_PATH' : artifact_path}).after(sentiment_server)
        
        
        
        news_reader_invok2 = funcs['func_invoke'].as_step(params = {"endpoint" : news_reader.outputs["endpoint"]},
                                                             handler="handler").after(news_reader)
    
    stocks_reader = funcs['stocks_reader'].deploy_step(env={'STOCK_LIST': STOCK_LIST,
                                                            'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_TSDB_TABLE': STOCKS_TSDB_TABLE,
                                                            'STOCKS_KV_TABLE': STOCKS_KV_TABLE,
                                                            'EXPRESSION_TEMPLATE': EXPRESSION_TEMPLATE,
                                                            'PROJECT_NAME' : project_name,
                                                            'ARTIFACT_PATH' : artifact_path})
    
    stream_viewer = funcs['stream_viewer'].deploy_step(env={'V3IO_CONTAINER': V3IO_CONTAINER,
                                                            'STOCKS_STREAM': STOCKS_STREAM}).after(news_reader_invok1,news_reader_invok2)
    
    
    
    vector_viewer = funcs['vector_reader'].deploy_step(env={'PROJECT_NAME' : project_name,
                                                            'ARTIFACT_PATH' : artifact_path}).after(stocks_reader,news_reader_invok1,news_reader_invok2)
    
    
    rnn_model_training_deployer = funcs["rnn_model_training"].deploy_step(env={'model_path': rnn_model_path,
                                                                               'PROJECT_NAME' : project_name,
                                                                               'ARTIFACT_PATH' : artifact_path})
    
    rnn_model_training_invoker = funcs['func_invoke'].as_step(params = {"endpoint" : rnn_model_training_deployer.outputs["endpoint"]},
                                                             handler="handler").after(rnn_model_training_deployer,vector_viewer)
    
    rnn_serving = funcs['rnn_serving'].deploy_step().after(rnn_model_training_invoker)
    
    rnn_model_prediction = funcs["rnn_model_prediction"].deploy_step(env = {"endpoint":rnn_serving.outputs['endpoint'],
                                                                            'ARTIFACT_PATH' : artifact_path}).after(rnn_serving)
    
    grafana_viewer = funcs["grafana_view"].deploy_step()
    
    grafana_viewer = funcs["grafana_view"].as_step(params = {"streamview_url" : stream_viewer.outputs["endpoint"],
                                                             "readvector_url" : vector_viewer.outputs["endpoint"],
                                                             "rnn_serving_url" : rnn_model_prediction.outputs["endpoint"],
                                                             "v3io_container" : V3IO_CONTAINER,
                                                             "stocks_kv" : STOCKS_KV_TABLE,
                                                             "stocks_tsdb" : STOCKS_TSDB_TABLE,
                                                             "grafana_url" : "http://grafana"},
                                                   handler = "handler").after(grafana_viewer,rnn_model_prediction)
