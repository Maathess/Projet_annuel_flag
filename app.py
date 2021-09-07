import os
import requests
from PIL import ExifTags, Image
import streamlit as st
import random


from ctypes import *
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import subprocess
import sys

REPO_DIR = 'https://github.com/Maathess/Projet_annuel_flag'

#st.set_page_config(page_title="Flag Predictor - Maathess", page_icon="🏴")

st.title("Flag predictor")

st.markdown('By <a href="https://github.com/Maathess/Projet_annuel_flag" target="_blank">Maathess</a>', unsafe_allow_html=True)

st.write("This project is about predicting flag using trained MLP models")

file_data = st.file_uploader("Select an image")

models = ["Sans couche cachée", "1 couche cachée, 8 neurones", "1 couche cachée, 32 neurones", "2 couches cachées, 32 neurones"]
option = st.selectbox(
    'Which models you want to try with?',
    models)

'You selected: ', option

if file_data is not None:
    with st.spinner('Classifying...'):
        # load the image from uploader; fix rotation for iOS devices if necessary
        img = file_data

        st.write('## Your Image')
        st.image(img)

        # classify
        if option == "Sans couche cachée" :
            st.write('## Model accuracy')
            st.image('./Train_Screenshoot/0couche_acc.PNG')

        elif option == "1 couche cachée, 8 neurones" :
            st.write('## Model accuracy')
            st.image('./Train_Screenshoot/1_8_acc.PNG')

        elif option == "1 couche cachée, 32 neurones" :
            st.write('## Model accuracy')
            st.image('./Train_Screenshoot/1_32_acc.PNG')

        elif option == "2 couches cachées, 32 neurones" :
            st.write('## Model accuracy')
            st.image('./Train_Screenshoot/2_32_acc.PNG')

        st.write('## What the model thinks')
        st.markdown(f'Model with "{option}" predicts that your image is a ... flag, correct ?')


