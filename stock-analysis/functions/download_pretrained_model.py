# Run this to download the pre-trained model to your `models` directory
import os


def download():
    
    url_prefix = os.environ.get('SAMPLE_DATA_SOURCE_URL_PREFIX', 'https://s3.wasabisys.com/iguazio/')
    model_location = f'{url_prefix}data/stock-analysis/model.pt'

    saved_models_directory = os.path.join(os.path.abspath('../tests/'), 'models')
    # Create paths
    os.makedirs(saved_models_directory, exist_ok=1)
    
    if "http" in model_location:
        command = "wget -nc -P {} {}".format(saved_models_directory,model_location)
    else:
        command = "cp {} {}".format(model_location, saved_models_directory)
        
    os.system(command)