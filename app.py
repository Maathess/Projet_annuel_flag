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

path_to_dll = "C:/Users/Maathess/Desktop/Projet_annuel_flag/PMC/cmake-build-debug/PMC9.dll"
mylib = cdll.LoadLibrary(path_to_dll)

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

def import_images_and_assign_labels(
        folder, label, X, Y
):
    image_path = os.path.join(folder, file)
    im = Image.open(image_path)
    im = im.resize((8, 8))
    im = im.convert("RGB")
    im_arr = np.array(im)
    im_arr = np.reshape(im_arr, (8 * 8 * 3))
    X.append(im_arr)
    Y.append(label)

def converte_img(conv_img):
        im = Image.open(conv_img)
        im = im.resize((8, 8))
        im = im.convert("RGB")
        im_arr = np.array(im)
        im_arr = np.reshape(im_arr, (8 * 8 * 3))
        X.append(im_arr)
        Y.append(label)



def import_dataset():
    dataset_folder = "C:/Users/Maathess/Desktop/Projet_annuel_flag/Datasets"
    train_folder = os.path.join(dataset_folder, "train")
    test_folder = os.path.join(dataset_folder, "test")

    X_train = []
    y_train = []
    import_images_and_assign_labels(
        os.path.join(train_folder, "brazil_flag"), [1, -1, -1], X_train, y_train
    )

    import_images_and_assign_labels(
        os.path.join(train_folder, "french_flag"), [-1, 1, -1], X_train, y_train
    )

    import_images_and_assign_labels(
        os.path.join(train_folder, "ireland_flag"), [-1, -1, 1], X_train, y_train
    )

    X_test = []
    y_test = []
    import_images_and_assign_labels(
        os.path.join(test_folder, "brazil_flag"), [1, -1, -1], X_test, y_test
    )

    import_images_and_assign_labels(
        os.path.join(test_folder, "french_flag"), [-1, 1, -1], X_test, y_test
    )

    import_images_and_assign_labels(
        os.path.join(test_folder, "ireland_flag"), [-1, -1, 1], X_test, y_test
    )

    return (np.array(X_train) / 255.0, np.array(y_train)), \
           (np.array(X_test) / 255.0, np.array(y_test))


def run_train(type_model, conv_img):
    (X_train, y_train), (X_test, y_test) = import_dataset()
    dataset_inputs = np.array(X_train)
    dataset_expected_outputs = np.array(y_train)

    if type_model == 0 :
        init_tab = [len(X_train[0]), 32, 3]
        init_size = len(init_tab)
        init_type = c_int * init_size
        init = init_type(*init_tab)

    elif type_model == 1 :
        init_tab = [len(X_train[0]), 32, 3]
        init_size = len(init_tab)
        init_type = c_int * init_size
        init = init_type(*init_tab)
    elif type_model == 2 :
        init_tab = [len(X_train[0]), 32, 3]
        init_size = len(init_tab)
        init_type = c_int * init_size
        init = init_type(*init_tab)
    elif type_model == 3 :
        init_tab = [len(X_train[0]), 32, 3]
        init_size = len(init_tab)
        init_type = c_int * init_size
        init = init_type(*init_tab)

    mylib.create_mlp_model.argtypes = [init_type, c_int]
    mylib.create_mlp_model.restype = c_void_p

    model_1_32 = mylib.create_mlp_model(init, int(init_size))
    test_dataset = X_test
    img_test = conv_img

    mylib.getXSize.argtypes = [c_void_p]
    mylib.getXSize.restype = c_int
    tmp_len = mylib.getXSize(model_1_32)

    flattened_dataset_inputs = []
    for p in dataset_inputs:
        for x in p:
            flattened_dataset_inputs.append(x)

    flattened_dataset_outputs = []
    for p in dataset_expected_outputs:
        flattened_dataset_outputs.append(p[0])
        flattened_dataset_outputs.append(p[1])
        flattened_dataset_outputs.append(p[2])

    # definition de train_classification_stochastic_gradient....
    arrsize_flat = len(flattened_dataset_inputs)
    arrtype_flat = c_float * arrsize_flat
    arr_flat = arrtype_flat(*flattened_dataset_inputs)

    arrsize_exp = len(flattened_dataset_outputs)
    arrtype_exp = c_float * arrsize_exp
    arr_exp = arrtype_exp(*flattened_dataset_outputs)

    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.argtypes = [c_void_p, arrtype_flat, c_int,
                                                                                         arrtype_exp, c_float, c_int]
    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model.restype = None

    mylib.train_classification_stochastic_gradient_backpropagation_mlp_model(model_1_32, arr_flat, arrsize_flat,
                                                                             arr_exp,
                                                                             0.0001, 10000)
    predicted_outputs = []
    for p in img_test:
        arrsizeP = len(p)
        arrtypeP = c_float * arrsizeP
        arrP = arrtypeP(*p)
        mylib.predict_mlp_model_classification.argtypes = [c_void_p, arrtypeP]
        mylib.predict_mlp_model_classification.restype = POINTER(c_float)

        tmp = mylib.predict_mlp_model_classification(model_1_32, arrP)
        np_arr = np.ctypeslib.as_array(tmp, (3,))
        predicted_outputs.append(np_arr)

    if np.argmax(predicted_outputs, axis=1) == 0:
        print("It's a brazil flag, right ?")

    elif np.argmax(predicted_outputs, axis=1) == 1:
        print("It's a french flag, right ?")
    elif np.argmax(predicted_outputs, axis=1) == 2:
        print("It's a ireland flag, right ?")
    else:
        print("You are joking with me ?")

REPO_DIR = 'https://github.com/Maathess/Projet_annuel_flag'

#st.set_page_config(page_title="Flag Predictor - Maathess", page_icon="🏴")

st.title("Flag predictor")

st.markdown('By <a href="https://github.com/Maathess/Projet_annuel_flag" target="_blank">Maathess</a>', unsafe_allow_html=True)

st.write("This project is about predicting flag using trained MLP models")

file_data = st.file_uploader("Select an image", type=["jpg"])
sans_couche = run_train(0)
une_couche_cachée_8_neurones = run_train(1)
une_couche_cachée_32_neurones = run_train(2)
deux_couches_cachées_32_neurones = run_train(3)
models = [sans_couche, une_couche_cachée_8_neurones, une_couche_cachée_32_neurones, deux_couches_cachées_32_neurones]
option = st.selectbox(
    'Which models you want to try with?',
    models)

'You selected: ', option

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

        converte_img(img)

        # classify
        if option == une_couche_cachée_8_neurones :
            une_couche_cachée_8_neurones

        elif option == une_couche_cachée_8_neurones :
            une_couche_cachée_8_neurones

        elif option == une_couche_cachée_32_neurones :
            une_couche_cachée_32_neurones

        elif option == deux_couches_cachées_32_neurones :
            deux_couches_cachées_32_neurones
'''            
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


