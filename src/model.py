import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import time


def drop_columns(df, lst):
    """
    Drops columns specified in the Pandas DataFrame
    :param df: DataFrame that will be altered
    :param lst: A list of strings, denoting columns in the DataFrame
    :return: New DataFrame, with specified columns removed.
    """
    df.drop(lst, axis=1, inplace=True)
    return df


def convert_to_factor(df, column):
    """
    Convert features to factors (categories) in H2OFrame
    :param df: Name of H2OFrame
    :param column: String of column name within H2OFrame
    :return: Column converted to factor
    """
    return df[column].asfactor()


if __name__ == '__main__':
    start = time.time()

    data = pd.read_csv("../data/Data_Entry_2017.csv")

    data.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                    'Patient_Age', 'Patient_Gender', 'View_Position',
                    'Original_Image_Width', 'Original_Image_Height',
                    'Original_Image_Pixel_Spacing_X',
                    'Original_Image_Pixel_Spacing_Y', 'Unnamed']

    data = drop_columns(data, ['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed',
                               'Original_Image_Width', 'Original_Image_Height', 'Image_Index'])

    # Check if number is in years, months, or days
    data['Age_Measure'] = data['Patient_Age'].astype(str).str[3]

    # Remove the character in Patient Age, and convert to integer
    data['Patient_Age'] = data['Patient_Age'].map(lambda x: str(x)[:-1]).astype(int)

    # Remove all categories right of the '|' delimiter in labels (preliminary category reduction)
    # Reduces categories from 709 to 15
    data['Finding_Labels'] = data['Finding_Labels'].apply(lambda x: x.split('|')[0])

    h2o.init(nthreads=-1)

    data = h2o.H2OFrame(data)

    # Convert features to factors (categorical)
    data['Follow_Up_#'] = convert_to_factor(data, "Follow_Up_#")
    data['Patient_ID'] = convert_to_factor(data, "Patient_ID")

    # print(data["Finding_Labels"].table().sort(by="Count", ascending=False))

    # Split Data
    print("Splitting Data")
    train, valid, test = data.split_frame(ratios=[0.6, 0.2], seed=7)

    # Train Model
    training_columns = [t for t in train.columns if t != 'Finding_Labels']

    # GBM CLASSIFIER
    print("Training GBM")
    gbm = H2OGradientBoostingEstimator(ntrees=100, distribution='multinomial', max_depth=4, learn_rate=0.01,
                                       balance_classes=True)
    gbm.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid)
    performance = gbm.model_performance(test_data=test)

    print("Logloss: ", gbm.logloss())
    print("RMSE: ", gbm.rmse())
    print("R2: ", gbm.r2())

    # aml=H2OAutoML(max_runtime_secs=10)
    # aml.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid, leaderboard_frame=test)

    y_pred = gbm.predict(test)

    print("Seconds: ", time.time() - start)
