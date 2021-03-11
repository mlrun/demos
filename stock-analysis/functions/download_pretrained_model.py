# Run this to download the pre-trained model to your `models` directory
import os


def download():
    model_location = 'https://iguazio-sample-data.s3.amazonaws.com/models/model.pt'
    saved_models_directory = os.path.join(os.path.abspath('../tests/'), 'models')
    # Create paths
    os.makedirs(saved_models_directory, exist_ok=1)
    command = "wget -nc -P {} {}".format(saved_models_directory,model_location)
    os.system(command)