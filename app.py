import os
import requests
from PIL import ExifTags, Image
import streamlit as st
import json
import random

REPO_DIR = 'https://github.com/Maathess/Projet_annuel_flag'
MODEL_FILE = 'dogs_online_resnet50_cpu.pkl'

#st.set_page_config(page_title="Flag Predictor - Maathess", page_icon="🏴")

st.title("Flag predictor")

st.markdown('By <a href="https://github.com/Maathess/Projet_annuel_flag" target="_blank">Maathess</a>', unsafe_allow_html=True)

st.write("This project is about predicting flag using trained MLP models")

file_data = st.file_uploader("Select an image", type=["jpg"])

model1 = [1,2,3,4]
option = st.selectbox(
    'Which models you want to try with?',
    model1)

'You selected: ', option

def download_file(url):
    with st.spinner('Downloading model...'):
        # from https://stackoverflow.com/a/16696317
        local_filename = url.split('/')[-1]
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk:
                    f.write(chunk)
        return local_filename

def fix_rotation(file_data):
    # check EXIF data to see if has rotation data from iOS. If so, fix it.
    try:
        image = Image.create(file_data)
    except (AttributeError, KeyError, IndexError):
        pass  # image didn't have EXIF data

    return image

'''
INFO MODEL
PLOT
ACCURACY ETC... '''
if file_data is not None:
    with st.spinner('Classifying...'):
        # load the image from uploader; fix rotation for iOS devices if necessary
        img = file_data

        st.write('## Your Image')
        st.image(img)
'''
        # classify
        pred, pred_idx, probs = learn.predict(img)
        top5_preds = sorted(list(zip(learn.dls.vocab, list(probs.numpy()))), key=lambda x: x[1], reverse=True)[:5]

        # prepare output
        out_text = '<table><tr> <th>Breed</th> <th>Confidence</th> <th>Example</th> </tr>'

        for pred in top5_preds:
            example = REPO_DIR + '/example_dogs/' + pred[0].replace(" ", "").lower() + ".jpg"
            out_text += '<tr>' + \
                        f'<td>{pred[0]}</td>' + \
                        f'<td>{100 * pred[1]:.02f}%</td>' + \
                        f'<td><img src="{example}" height="150" /></td>' + \
                        '</tr>'
        out_text += '</table><br><br>'

        st.write('## What the model thinks')
        st.markdown(out_text, unsafe_allow_html=True)
'''


