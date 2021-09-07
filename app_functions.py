import os
import requests
from PIL import ExifTags, Image
import streamlit as st
import random


from ctypes import *
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

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
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        im = Image.open(image_path)
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
