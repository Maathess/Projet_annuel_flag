import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow.keras as keras
from PIL import Image
import numpy as np

DATASET_FOLDER = "F:/2021_3A_IABD1_FakeDataset"
TRAIN_SUBFOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_SUBFOLDER = os.path.join(DATASET_FOLDER, "test")

TRAIN_PORTE_FOLDER = os.path.join(TRAIN_SUBFOLDER, "porte")
TRAIN_FENETRE_FOLDER = os.path.join(TRAIN_SUBFOLDER, "fenetre")

TEST_PORTE_FOLDER = os.path.join(TEST_SUBFOLDER, "porte")
TEST_FENETRE_FOLDER = os.path.join(TEST_SUBFOLDER, "fenetre")


def fill_x_and_y_with_images_and_labels(folder, x_list, y_list, label):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        image = Image.open(file_path)
        image = image.resize((32, 32))
        im_arr = np.array(image).flatten()
        im_arr = im_arr / 255.0
        x_list.append(im_arr)
        y_list.append(label)


def import_dataset():
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    fill_x_and_y_with_images_and_labels(TRAIN_PORTE_FOLDER, X_train, Y_train, 1)
    fill_x_and_y_with_images_and_labels(TRAIN_FENETRE_FOLDER, X_train, Y_train, 0)

    fill_x_and_y_with_images_and_labels(TEST_PORTE_FOLDER, X_test, Y_test, 1)
    fill_x_and_y_with_images_and_labels(TEST_FENETRE_FOLDER, X_test, Y_test, 0)

    return (np.array(X_train).astype(np.float), np.array(Y_train).astype(np.float)), \
           (np.array(X_test).astype(np.float), np.array(Y_test).astype(np.float))


def run():
    (X_train, Y_train), (X_test, Y_test) = import_dataset()

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))

    model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.mse)

    model.fit(X_train, Y_train, epochs=50)

    print(model.predict(X_train))
    print(model.predict(X_test))


if __name__ == "__main__":
    run()