from imutils import paths
import pandas as pd
import v3io_frames as v3f


def handler(context, event):
    data_path = '/User/demos/demos/faces/dataset/'
    artifact_path = 'User/demos/demos/faces/artifacts/'

    classes_url = artifact_path + 'idx2name.csv'
    classes_df = pd.read_csv(classes_url)

    known_classes = [n.replace('_', ' ') for n in classes_df['name'].values]
    options = ['None'] + known_classes + ['add new employee', 'not an employee']

    images = [f for f in paths.list_images(data_path) if '.ipynb' not in f]

