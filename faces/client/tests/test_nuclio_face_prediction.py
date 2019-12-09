import nuclio_sdk

import cv2
import face_recognition
import imutils
import joblib
import json
import numpy as np
import pandas as pd
import random
import string
import v3io_frames as v3f
import os
import datetime
import logging
import shutil


class SKModel(object):
    def __init__(self):
        self.name = 'model.bst'
        self.model_filepath = os.environ['MODEL_PATH']
        self.model = None
        self.ready = None
        self.classes = os.environ['CLASSES_MAP']
        self.dataset_path = os.environ['DATASET_PATH']

    def move_unknown_file(self,new_row, img_url):
        destination = self.unknown_path + '/' + img_url.split('/')[-1]
        new_row['imgUrl'] = destination
        dest = shutil.move(img_url, destination)
        return destination


    def load(self):
        self.model = joblib.load(self.model_filepath)
        self.ready = True

    def predict(self, context, data):

        # acquires all metadata
        time = data['time']
        cam_name = data['camera']
        img_url = data['file_path']

        # prepares image for use
        with open(img_url, 'rb') as f:
            content = f.read()
        img_bytes = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(img_bytes, flags=1)

        # converts image format to RGB for comptability with face_recognition library and resize for faster processing
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(image, width=750)
        ratio = image.shape[1] / float(rgb.shape[1])

        # gets mapping from label to name and known encodings
        idx2name_df = pd.read_csv(self.classes).set_index('value')

        if not self.model:
            self.load()

        # locates faces in image and extracts embbeding vector for each face
        context.logger.info_with('[INFO]', msg="recognizing faces...")
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        # determines if face is a clear match/ambiguous.
        names = []
        labels = []
        for encoding in encodings:
            name = 'unknown'
            label = 'unknown'
            probs = self.model.predict_proba(encoding.reshape(1, -1))
            if np.max(probs) > 0.5:
                label = np.argmax(probs)
                name = idx2name_df.loc[label]['name'].replace('_', ' ')
            names.append(name)
            labels.append(label)
            context.logger.info(f'{name} with {probs}')

        # frames client to save necessary data
        client = v3f.Client("framesd:8081", container="users")

        # draw boxes with name on the image and performs logic according to match/ambiguous
        ret_list = []
        for ((top, right, bottom, left), name, encoding, label) in zip(boxes, names, encodings, labels):

            # rescale the face coordinates
            top = int(top * ratio)
            right = int(right * ratio)
            bottom = int(bottom * ratio)
            left = int(left * ratio)

            # random string for unique name in our saved data
            rnd_tag = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

            new_row = {}
            # saves all extracted data to kv.
            new_row = {'c' + str(i).zfill(3): encoding[i] for i in range(128)}
            if (
                    name != 'unknown'):  # and (len(enc_df.loc[enc_df['label'] == label]) < 50): TODO add different logic for limit of images of same person in dataset
                new_row['label'] = label
            else:
                new_row['label'] = -1
            new_row['imgUrl'] = img_url
            new_row['fileName'] = name.replace(' ', '_') + '_' + rnd_tag
            new_row['camera'] = cam_name
            new_row['time'] = datetime.datetime.utcnow()

            if new_row['label'] == -1:
                self.move_unknown_file(new_row, img_url)
                context.logger.debug('moving ' + img_url + 'to' + destination)
                destination = self.unknown_path+'/'+img_url.split('/')[-1]
                new_row['imgUrl'] = destination
                dest = shutil.move(img_url, destination)
                context.logger.debug('moving ' + img_url +'to' +destination)

            new_row_df = pd.DataFrame(new_row, index=[0])

            client.write(backend='kv', table='iguazio/demos/demos/faces/artifacts/encodings', dfs=new_row_df,
                         index_cols=['fileName'])


            # appends box and name to the returned list
            ret_list.append(((top, right, bottom, left), name, np.max(probs)))

        return ret_list


def handler(context, event):
    model = SKModel()
    return model.predict(context=context, data=event.body)

os.environ['MODEL_PATH']='model.bst'
os.environ['CLASSES_MAP']='idx2name.csv'

logger = logging.Logger('sahar')
ctx = nuclio_sdk.Context(logger=logger)

payload = {'file_path': 'images/image.jpg', 'time': '20191110131130', 'camera': 'cammy'}
ev = nuclio_sdk.Event(body=payload)
handler(ctx,ev)