import os

import pandas as pd


def get_lst_images(file_path):
    """
    Reads in all files from file path into a list.

    INPUT
        file_path: specified file path containing the images.

    OUTPUT
        List of image strings
    """
    return [i for i in os.listdir(file_path) if i != '.DS_Store']


if __name__ == '__main__':
    data = pd.read_csv("../data/Data_Entry_2017.csv")
    sample = os.listdir('../data/resized-512/')

    sample = pd.DataFrame({'Image Index': sample})

    sample = pd.merge(sample, data, how='left', on='Image Index')

    sample.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                      'Patient_Age', 'Patient_Gender', 'View_Position',
                      'Original_Image_Width', 'Original_Image_Height',
                      'Original_Image_Pixel_Spacing_X',
                      'Original_Image_Pixel_Spacing_Y', 'Unnamed']

    sample['Finding_Labels'] = sample['Finding_Labels'].apply(lambda x: x.split('|')[0])

    sample.drop(['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed'], axis=1, inplace=True)
    sample.drop(['Original_Image_Width', 'Original_Image_Height'], axis=1, inplace=True)

    print("Writing CSV")
    sample.to_csv('../data/sample_labels.csv', index=False, header=True)
