import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator


def drop_columns(df, lst):
    """
    Drops columns specified in the Pandas DataFrame
    :param df: DataFrame that will be altered
    :param lst: A list of strings, denoting columns in the DataFrame
    :return: New DataFrame, with specified columns removed.
    """
    df.drop(lst, axis=1, inplace=True)
    return df


def get_training_columns(df, target):
    """
    Creates list of columms for training (h2o only)
    :param df: H2OFrame to be used for training
    :param target: Target column within training H2OFrame
    :return: list of strings, containig all training columns
    """
    return [col for col in df.columns if col != target]


if __name__ == '__main__':

    data = pd.read_csv("../data/Data_Entry_2017.csv", skiprows=1, names=['Image_Index', 'Finding_Labels', 'Follow_Up_#',
                                                                         'Patient_ID', 'Patient_Age', 'Patient_Gender',
                                                                         'View_Position', 'Original_Image_Width',
                                                                         'Original_Image_Height',
                                                                         'Original_Image_Pixel_Spacing_X',
                                                                         'Original_Image_Pixel_Spacing_Y',
                                                                         'Unnamed'], low_memory=False)

    data = drop_columns(data, ['Follow_Up_#', 'Unnamed', 'Original_Image_Width',
                               'Original_Image_Height', 'Image_Index', 'Patient_ID',
                               'Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y'])


    # Remove the character in Patient Age, and convert to integer
    data['Patient_Age'] = data['Patient_Age'].map(lambda x: str(x)[:-1]).astype(int)

    # Remove all categories right of the first '|' delimiter in labels
    # Reduces categories from 709 to 15
    data['Finding_Labels'] = data['Finding_Labels'].apply(lambda x: x.split('|')[0])

    h2o.init()

    data = h2o.H2OFrame(data)

    train, valid, test = data.split_frame(ratios=[0.6, 0.2], seed=8)

    training_columns = get_training_columns(train, "Finding_Labels")

    gbm = H2OGradientBoostingEstimator(ntrees=1000, distribution='multinomial', max_depth=2, learn_rate=0.001,
                                       balance_classes=True
                                       # stopping_metric="logloss",
                                       )

    gbm.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid)
    # performance = gbm.model_performance(test_data=test)

    # h2o.cluster().shutdown()
