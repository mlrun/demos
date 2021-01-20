from kfp import dsl
from mlrun import mount_v3io

funcs = {}


def init_functions(functions: dict, project=None, secrets=None):
    '''
    This function will run before running the project.
    It allows us to add our specific system configurations to the functions
    like mounts or secrets if needed.

    In this case we will add Iguazio's user mount to our functions using the
    `mount_v3io()` function to automatically set the mount with the needed
    variables taken from the environment. 
    * mount_v3io can be replaced with mlrun.platforms.mount_pvc() for 
    non-iguazio mount

    @param functions: <function_name: function_yaml> dict of functions in the
                        workflow
    @param project: project object
    @param secrets: secrets required for the functions for s3 connections and
                    such
    '''
    for f in functions.values():
        f.apply(mount_v3io())                  # On Iguazio (Auto-mount /User)
        # f.apply(mlrun.platforms.mount_pvc()) # Non-Iguazio mount
        
    functions['serving'].set_env('MODEL_CLASS', 'TFModel')
    functions['serving'].set_env('IMAGE_HEIGHT', '224')
    functions['serving'].set_env('IMAGE_WIDTH', '224')
    functions['serving'].set_env('ENABLE_EXPLAINER', 'False')
    functions['serving'].spec.min_replicas = 1


@dsl.pipeline(
    name='Image classification demo',
    description='Train an Image Classification TF Algorithm using MLRun'
)
def kfpipeline(
        image_archive='store:///images',
        images_dir='/User/artifacts/images',
        checkpoints_dir='/User/artifacts/models/checkpoints',
        model_name='cat_vs_dog_tfv1',
        epochs=2):

    # step 1: download and prep images
    open_archive = funcs['utils'].as_step(name='download',
                                          handler='open_archive',
                                          params={'target_path': images_dir},
                                          inputs={'archive_url': image_archive},
                                          outputs=['content'])

    # step 2: train the model
    train_dir = str(open_archive.outputs['content']) + '/train'
    val_dir = str(open_archive.outputs['content']) + '/validation'
    train = funcs['trainer'].as_step(name='train',
                                     params={'epochs': epochs,
                                             'checkpoints_dir': checkpoints_dir,
                                             'model_dir'     : 'tfmodels',
                                             'train_path'     : train_dir,
                                             'val_path'       : val_dir,
                                             'batch_size'     : 32},
                                     outputs=['model'])

    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(models={model_name: train.outputs['model']})
