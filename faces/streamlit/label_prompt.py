import streamlit as st
import cv2
from imutils import paths
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import datetime
import time


DATA_PATH = 'images'  # '/User/face_recognition/dataset/'
artifact_path = ''  # '/User/face_recognition/artifacts/'
classes_url = artifact_path + 'idx2name.csv'

st.title('Label Unknown Images')

# generates list of valid labeling options
classes_df = pd.read_csv(classes_url)
known_classes = [n.replace('_', ' ') for n in classes_df['name'].values]
options = ['None'] + known_classes + ['add new employee', 'not an employee']

def load_images(data_path):
    return [f for f in paths.list_images(data_path) if '.ipynb' not in f]



if __name__ == '__main__':
    images = load_images(DATA_PATH)

    try:
        # Are there any images left to tag?
        # - Yes
        if len(images) > 0:
            # Show image select box
            st.subheader('Do you know this person?')
            idx = st.selectbox('Choose picture to label', range(len(images)))

            selected_label = st.radio(label='Select label for the image',
                                      options=options,
                                      key=0)


        path = images[idx]
        img = cv2.imread(path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.axis('off')
        st.pyplot()

        # Was a label selected (in previous step) ?
        # - Yes
        if selected_label != "None":
                tag_time = datetime.datetime.now()
                date_time = tag_time.strftime("%d:%m:%Y:%H%M%S")
                if selected_label == 'not an employee':
                    file_name = 'unknowns/' + date_time + '.jpg'
                elif selected_label == 'add new employee':
                    new_name = st.text_input('Please enter name of employee')
                    dir_name = 'unprocessed/' + new_name.replace(' ', '_')
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    file_name = dir_name + '/' + date_time + '.jpg'
                else:
                    dir_name = 'unprocessed/' + selected_label.replace(' ', '_')
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    file_name = dir_name + '/' + date_time + '.jpg'

                    shutil.move(path, file_name)
                st.write(selected_label)
    except:
        st.success('No more images to label')

    # i= 0
    # done = None
    # if len(images) > 0:
    #     imagePaths = [f for f in paths.list_images(DATA_PATH) if '.ipynb' not in f]
    #     st.subheader('Image labeled successfully')
    # else:

# streamlit run label_prompt.py