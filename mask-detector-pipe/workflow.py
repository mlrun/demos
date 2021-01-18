from kfp import dsl
import mlrun

funcs = {}
epochs = 70
replicas = 2
batch_size = 32

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

    :param functions:  <function_name: function_yaml> dict of functions in the
                        workflow
    :param project:    project object
    :param secrets:    secrets required for the functions for s3 connections and
                       such
    '''
    for f in functions.values():
        f.apply(mlrun.mount_v3io())            # On Iguazio (Auto-mount /User)
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
    # setup pipeline params
        image_archive='store:///images',
        target_path='/User/artifacts/fm-images',
        checkpoints_dir='/User/artifacts/models/checkpoints',
        model_name='mask_vs_no_mask'):

    # step 1: download images
    open_archive = funcs['utils'].as_step(name='download_and_open',
                                          handler='open_archive',
                                          params={'target_path': target_path},
                                          inputs={'archive_url': image_archive},
                                          outputs=['content'])

    # step 2: deploy out serveless training function and train a model
    # get the output of step 1 and use it as source dir
    source_dir = str(open_archive.outputs['content'])
    source_dir_images = source_dir +'/images'
    source_dir_annot = source_dir +'/annotations'
    
    # deploy our trainer function 
    deploy_train = funcs['trainer'].deploy_step(skip_deployed=True)
    deploy_train.after(open_archive)
    
    # train our model
    train = funcs['trainer'].as_step(name='train',
                                     params={'epochs'         : epochs,
                                             'batch_size'     : batch_size,
                                             'imgs'           : source_dir_images,
                                             'annot'          : source_dir_annot,
                                             'model_artifacts': checkpoints_dir},
                                     outputs=['model'],
                                     image=deploy_train.outputs['image'])
    
    # the image_pull_policy param is set in order to use the image 
    # we just built in the cell above
    train.container.set_image_pull_policy('Always')
    
    # set timeout in case the pipeline takes time
    train.set_timeout(214748364)
    train.after(deploy_train)
    
    # step 3: deploy the model using nuclio functions
    deploy = funcs['serving'].deploy_step(models={model_name: train.outputs['model']})
    deploy.after(train)
