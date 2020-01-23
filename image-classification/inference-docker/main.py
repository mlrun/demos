import json
import os
import numpy as np
import requests
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from os import environ, path
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

class TFModel(object):
    def __init__(self, name: str, model_dir: str):
        self.name = name
        self.model_filepath = model_dir
        self.model = None
        self.ready = None

        self.IMAGE_WIDTH = int(environ['IMAGE_WIDTH'])
        self.IMAGE_HEIGHT = int(environ['IMAGE_HEIGHT'])
        
        try:
            with open(environ['classes_map'], 'r') as f:
                self.classes = json.load(f)
        except:
            self.classes = None
        
        print(f'Classes: {self.classes}')

    def load(self):
        self.model = load_model(self.model_filepath)

        self.ready = True

    def _download_file(self, url, target_path):
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    def predict(self, context, data):
        try:
            img = Image.open(BytesIO(data))
            img = img.resize((self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

            # Load image
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Predict
            predicted_probability = self.model.predict(images)

            # return prediction
            if self.classes:
                predicted_classes = np.around(predicted_probability, 1).tolist()[0]
                predicted_probabilities = predicted_probability.tolist()[0]
                print(predicted_classes)
                print(predicted_probabilities)
                return {
                    'prediction': [self.classes[str(int(cls))] for cls in predicted_classes], 
                    f'{self.classes["1"]}-probability': predicted_probabilities
                }
            else:
                return predicted_probability.tolist()[0]

        except Exception as e:
            raise Exception("Failed to predict {}".format(e))

def predict(context, model_name, event):
    global models
    global protocol

    # Load the requested model
    model = models[model_name]

    # Verify model is loaded (Async)
    if not model.ready:
        model.load()
    
    # extract image data from event
    try:
        data = event.body
        ctype = event.content_type
        if not ctype or ctype.startswith('text/plain'):
            # Get image from URL
            url = data.decode('utf-8')
            context.logger.debug_with('downloading image', url=url)
            data = urlopen(url).read()
            
    except Exception as e:
        raise Exception("Failed to get data: {}".format(e))                
            
    # Predict
    results = model.predict(context, data)
    context.logger.info(results)

    # Wrap & return response
    return context.Response(body=json.dumps(results),
                            headers={},
                            content_type='text/plain',
                            status_code=200)

# Router
paths = {
    'predict': predict,
    'explain': '',
    'outlier_detector': '',
    'metrics': '',
}

# Definitions
model_prefix = 'SERVING_MODEL_'
models = {}

def init_context(context):
    global models
    global model_prefix

    # Initialize models from environment variables
    # Using the {model_prefix}_{model_name} = {model_path} syntax
    model_paths = {k[len(model_prefix):]: v for k, v in os.environ.items() if
                   k.startswith(model_prefix)}

    models = {name: TFModel(name=name, model_dir=path) for name, path in
              model_paths.items()}
    context.logger.info(f'Loaded {list(models.keys())}')

err_string = 'Got path: {}\nPath must be <host>/<action>/<model-name> \nactions: {} \nmodels: {}'

def handler(context, event):
    global models
    global paths

    # check if valid route & model
    sp_path = event.path.strip('/').split('/')
    if len(sp_path) < 2 or sp_path[0] not in paths or sp_path[1] not in models:
        return context.Response(body=err_string.format(event.path, '|'.join(paths), '|'.join(models.keys())),
                                content_type='text/plain',
                                status_code=400)
        
    function_path = sp_path[0] 
    model_name = sp_path[1]

    context.logger.info(
        f'Serving uri: {event.path} for route {function_path} '
        f'with {model_name}, content type: {event.content_type}')

    route = paths.get(function_path)
    if route:
        return route(context, model_name, event)

    return context.Response(body='function {} not implemented'.format(function_path),
                            content_type='text/plain',
                            status_code=400)