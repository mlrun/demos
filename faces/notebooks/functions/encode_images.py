from mlrun import get_or_create_ctx
import torch
import os
import shutil
import zipfile
from urllib.request import urlopen
from io import BytesIO
import face_recognition
from imutils import paths
import cv2
from mlrun.artifacts.dataset import TableArtifact
import pandas as pd
import datetime
import random
import string
import v3io_frames as v3f
from functions.params import Params


def encode_images(context):
    params = Params()
    params.set_params_from_context(context)
    context.logger.info(params)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    context.logger.info(f'Running on device: {device}')

    client = v3f.Client(params.frames_url, container="faces", token=params.token)

    if not os.path.exists(params.data_path + 'processed'):
        os.makedirs(params.data_path + 'processed')

    if not os.path.exists(params.data_path + 'label_pending'):
        os.makedirs(params.data_path + 'label_pending')

    # If no train images exist in the predefined path we will train the model on a small dataset of movie actresses
    if not os.path.exists(params.data_path + 'input'):
        os.makedirs(params.data_path + 'input')
        resp = urlopen('https://iguazio-public.s3.amazonaws.com/roy-actresses/Actresses.zip')
        zip_ref = zipfile.ZipFile(BytesIO(resp.read()), 'r')
        zip_ref.extractall(params.data_path + 'input')
        zip_ref.close()

    if os.path.exists(params.data_path + 'input/__MACOSX'):
        shutil.rmtree(params.data_path + 'input/__MACOSX')

    idx_file_path = params.artifacts_path + "idx2name.csv"
    if os.path.exists(idx_file_path):
        idx2name_df = pd.read_csv(idx_file_path)
    else:
        idx2name_df = pd.DataFrame(columns=['value', 'name'])

    # creates a mapping of classes(person's names) to target value
    new_classes_names = [f for f in os.listdir(params.data_path + 'input') if
                         not '.ipynb' in f and f not in idx2name_df['name'].values]

    initial_len = len(idx2name_df)
    final_len = len(idx2name_df) + len(new_classes_names)
    for i in range(initial_len, final_len):
        idx2name_df.loc[i] = {'value': i, 'name': new_classes_names.pop()}

    name2idx = idx2name_df.set_index('name')['value'].to_dict()

    # log name to index mapping into mlrun context
    context.log_artifact(TableArtifact('idx2name', df=idx2name_df), local_path='idx2name.csv')

    # generates a list of paths to labeled images
    imagePaths = [f for f in paths.list_images(params.data_path + 'input') if not '.ipynb' in f]
    knownEncodings = []
    knownLabels = []
    fileNames = []
    urls = []
    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        # extracts label (person's name) of the image
        name = imagePath.split(os.path.sep)[-2]

        # prepares to relocate image after extracting features
        file_name = imagePath.split(os.path.sep)[-1]
        new_path = params.data_path + 'processed/' + file_name

        # converts image format to RGB for comptability with face_recognition library
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detects coordinates of faces bounding boxes
        boxes = face_recognition.face_locations(rgb, model='hog')

        # computes embeddings for detected faces
        encodings = face_recognition.face_encodings(rgb, boxes)

        # this code assumes that a person's folder in the dataset does not contain an image with a face other then his own
        for enc in encodings:
            file_name = name + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            knownEncodings.append(enc)
            knownLabels.append([name2idx[name]])
            fileNames.append(file_name)
            urls.append(new_path)

        # move image to processed images directory
        shutil.move(imagePath, new_path)

    # saves computed encodings to avoid repeating computations
    df_x = pd.DataFrame(knownEncodings, columns=['c' + str(i).zfill(3) for i in range(128)]).reset_index(drop=True)
    df_y = pd.DataFrame(knownLabels, columns=['label']).reset_index(drop=True)
    df_details = pd.DataFrame([['initial training'] * 3] * len(df_x), columns=['imgUrl', 'camera', 'time'])
    df_details['time'] = [datetime.datetime.utcnow()] * len(df_x)
    df_details['imgUrl'] = urls
    data_df = pd.concat([df_x, df_y, df_details], axis=1)
    data_df['fileName'] = fileNames

    client.write(backend='kv', table='encodings', dfs=data_df,
                 index_cols=['fileName'])

    encoding_path = "encoding"
    # with open('encodings_path.txt', 'w+') as f:
    #     f.write('encodings')
    context.log_artifact('encodings_path', body = encoding_path)
    #os.remove('encodings_path.txt')


if __name__ == '__main__':
    context = get_or_create_ctx('encoding')
    encode_images(context)
