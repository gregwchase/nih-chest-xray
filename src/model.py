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


if __name__ == '__main__':
    start = time.time()

    data = pd.read_csv("../data/Data_Entry_2017.csv")

    data.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                    'Patient_Age', 'Patient_Gender', 'View_Position',
                    'Original_Image_Width', 'Original_Image_Height',
                    'Original_Image_Pixel_Spacing_X',
                    'Original_Image_Pixel_Spacing_Y', 'Unnamed']

    # Drop columns that aren't needed.
    # data.drop(['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed'], axis=1, inplace=True)
    # data.drop(['Original_Image_Width', 'Original_Image_Height'], axis=1, inplace=True)

    data = drop_columns(data,['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed', 'Original_Image_Width', 'Original_Image_Height', 'Image_Index'])

    # Check if number is in years, months, or days
    data['Age_Measure'] = data['Patient_Age'].astype(str).str[3]

    # Remove the character in Patient Age, and convert to integer
    data['Patient_Age'] = data['Patient_Age'].map(lambda x: str(x)[:-1]).astype(int)

    # Remove everything to the right of the '|' delimiter in the labels (preliminary category reduction)
    # Reduce categories from 709 to 15
    data['Finding_Labels'] = data['Finding_Labels'].apply(lambda x: x.split('|')[0])

    print("Initializing H2O Cluster")
    h2o.init(nthreads=-1)

    data = h2o.H2OFrame(data)



    # Convert features to factors (categorical)
    data['Follow_Up_#'] = data['Follow_Up_#'].asfactor()
    data['Patient_ID'] = data['Patient_ID'].asfactor()

    print(data["Finding_Labels"].table().sort(by="Count", ascending=False))

    # Split Data
    print("Splitting Data")
    split = data.split_frame(ratios=[0.6, 0.2], seed=7)

    train = split[0]
    valid = split[1]
    test = split[2]

    # Train Model
    training_columns = [t for t in train.columns if t != 'Finding_Labels']

    # GBM CLASSIFIER
    print("Training GBM")
    gbm = H2OGradientBoostingEstimator(ntrees=200, distribution='multinomial', max_depth=4, learn_rate=0.1, balance_classes=True)
    gbm.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid)
    performance = gbm.model_performance(test_data=test)

    print("Logloss: ", gbm.logloss())
    print("RMSE: ", gbm.rmse())
    print("R2: ", gbm.r2())

    # aml=H2OAutoML(max_runtime_secs=10)
    # aml.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid, leaderboard_frame=test)

    y_pred = gbm.predict(test)

    print("Seconds: ", time.time() - start)
