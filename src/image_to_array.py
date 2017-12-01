import time

import cv2
import numpy as np
import pandas as pd


def convert_images_to_arrays(file_path, df):
    """
    Converts each image to an array, and appends each array to a new NumPy
    array, based on the image column equaling the image file name.

    INPUT
        file_path: Specified file path for resized test and train images.
        df: Pandas DataFrame being used to assist file imports.

    OUTPUT
        NumPy array of image arrays.
    """

    lst_imgs = [l for l in df['Image_Index']]

    return np.array([np.array(cv2.imread(file_path + img, cv2.IMREAD_GRAYSCALE)) for img in lst_imgs])


def save_to_array(arr_name, arr_object):
    """
    Saves data object as a NumPy file. Used for saving train and test arrays.

    INPUT
        arr_name: The name of the file you want to save.
            This input takes a directory string.
        arr_object: NumPy array of arrays. This object is saved as a NumPy file.
    """
    return np.save(arr_name, arr_object)


if __name__ == '__main__':
    start_time = time.time()

    labels = pd.read_csv("../data/sample_labels.csv")

    print("Writing Train Array")
    X_train = convert_images_to_arrays('../data/resized-512/', labels)

    print(X_train.shape)

    print("Saving Train Array")
    save_to_array('../data/X_sample.npy', X_train)

    print("Seconds: ", round(time.time() - start_time), 2)
