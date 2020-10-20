class Params(object):
    def __init__(self
                 , data_path='dataset/'
                 , artifacts_path='artifacts/'
                 , models_path='/User/mlrun/demos/faces/notebooks/functions/models.py'
                 , frames_url='framesd:8081'
                 , token='set_token'
                 ):
        self.data_path = data_path
        self.artifacts_path = artifacts_path
        self.models_path = models_path
        self.frames_url = frames_url
        self.token = token

    def set_params_from_context(self, context):
        context.logger.info("setting context params")
        attrs = vars(self)
        for attr in attrs:
            if attr in context.parameters:
                setattr(self, attr, context.parameters[attr])
        attrs = vars(self)
        context.logger.info(attrs)
