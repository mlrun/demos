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
    functions['serving'].set_env('IMAGE_HEIGHT', '128')
    functions['serving'].set_env('IMAGE_WIDTH', '128')
    functions['serving'].set_env('ENABLE_EXPLAINER', 'False')


@dsl.pipeline(
    name='Image classification demo',
    description='Train an Image Classification TF Algorithm using MLRun'
)
def kfpipeline(
        image_archive='store:///images',
        images_path='/User/mlrun/examples/images',
        source_dir='/User/mlrun/examples/images/cats_n_dogs',
        checkpoints_dir='/User/mlrun/examples/checkpoints',
        model_path='models/cats_n_dogs.h5',
        model_name='cat_vs_dog_v1'):

    # step 1: download images
    open_archive = funcs['utils'].as_step(name='download',
                                          handler='open_archive',
                                          out_path=images_path,
                                          params={'target_dir': images_path},
                                          inputs={'archive_url': image_archive},
                                          outputs=['content'])

    # step 2: label images
    label = funcs['utils'].as_step(name='label',
                                   handler='categories_map_builder',
                                   out_path=images_path,
                                   params={'source_dir': source_dir},
                                   outputs=['categories_map',
                                            'file_categories']).after(open_archive)

    # step 3: train the model
    train = funcs['trainer'].as_step(name='train',
                                     params={'epochs': 8,
                                             'checkpoints_dir': checkpoints_dir,
                                             'model_path'     : model_path,
                                             'data_path'      : source_dir,
                                             'batch_size'     : 224},
                                     inputs={
                                         'categories_map': label.outputs['categories_map'],
                                         'file_categories': label.outputs['file_categories']},
                                     outputs=['model'])
    train.container.set_image_pull_policy('Always')

    # deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(models={model_name: train.outputs['model']})
