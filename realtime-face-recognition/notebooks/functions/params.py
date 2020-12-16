class Params(object):
    def __init__(self
                 , data_path='faces/dataset/'
                 , artifacts_path='faces/artifacts/'
                 , models_path='/User/mlrun/demos/faces/notebooks/functions/models.py'
                 , frames_url='framesd:8081'
                 , token='set_token'
                 , encodings_path='faces/encodings/'
                 , container='users'
                 ):
        self.data_path = data_path
        self.artifacts_path = artifacts_path
        self.models_path = models_path
        self.frames_url = frames_url
        self.token = token
        self.encodings_path = encodings_path
        self.container = container

    def set_params_from_context(self, context):
        context.logger.info("setting context params")
        attrs = vars(self)
        for attr in attrs:
            if attr in context.parameters:
                setattr(self, attr, context.parameters[attr])
        attrs = vars(self)
        context.logger.info(attrs)
