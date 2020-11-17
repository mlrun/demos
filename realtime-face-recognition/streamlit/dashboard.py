import streamlit as st
import cv2
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import v3io_frames as v3f
import os
import shutil
import datetime


def load_images(data_path):
    return [f for f in paths.list_images(data_path) if '.ipynb' not in f]


@st.cache
def load_enc_df():
    return client.read(backend="kv", table='iguazio/demos/demos/realtime-face-recognition/artifacts/encodings', reset_index=True, filter="label!=-1")


if __name__ == '__main__':
    client = v3f.Client("framesd:8081", container="users")
    data_path = '/User/demos/demos/realtime-face-recognition/dataset/'
    artifact_path = 'User/demos/demos/realtime-face-recognition/artifacts/'
    classes_url = artifact_path + 'idx2name.csv'
    classes_df = pd.read_csv(classes_url)
    known_classes = [n.replace('_', ' ') for n in classes_df['name'].values]

    page = st.sidebar.selectbox('Choose option', ['Label Unknown Images', 'View Collected Images'], key=1)
    if page == 'Label Unknown Images':

        images = load_images(data_path + 'label_pending')
        st.title('Label Unknown Images')

        # generates list of valid labeling options
        options = ['None'] + known_classes + ['add new employee', 'not an employee']

        # Are there any images left to tag?
        # - Yes
        if len(images) > 0:
            # Show image select box
            idx = st.selectbox('Choose picture to label', range(len(images)))
            path = images[idx]
            img = cv2.imread(path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            st.subheader('Do you know this person?')
            plt.imshow(rgb_img)
            plt.axis('off')
            st.pyplot()

            selected_label = st.selectbox(label='Select label for the image', options=options, key=0)

            # Was a label selected (in previous step) ?
            # - Yes
            if selected_label != "None":
                tag_time = datetime.datetime.now()
                date_time = tag_time.strftime("%d:%m:%Y:%H%M%S")
                if selected_label == 'not an employee':
                    dir_name = data_path + 'unrecognized'
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    file_name = dir_name + '/' + date_time + '.jpg'
                elif selected_label == 'add new employee':
                    new_name = st.text_input('Please enter name of employee')
                    dir_name = data_path + 'input/' + new_name.replace(' ', '_')
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    file_name = dir_name + '/' + date_time + '.jpg'
                else:
                    dir_name = data_path + 'input/' + selected_label.replace(' ', '_')
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    file_name = dir_name + '/' + date_time + '.jpg'
                if st.button('apply', key=100):
                    shutil.move(path, file_name)
                    st.empty()
        else:
            st.success('No more images to label')

    if page == 'View Collected Images':
        st.title('View Collected Images')
        enc_df = load_enc_df()
        view_df = enc_df[['fileName', 'camera', 'time']]
        view_df = view_df.rename(columns={'fileName': 'identifier'})
        view_df['identifier'] = view_df['identifier']
        st.dataframe(view_df)

        idx = st.selectbox('Choose image to view', range(len(view_df)), key=2)

        img_url = enc_df.iloc[idx]['imgUrl']
        kv_img = cv2.imread(img_url)
        rgb_kv_img = cv2.cvtColor(kv_img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_kv_img)
        plt.axis('off')
        st.pyplot()
