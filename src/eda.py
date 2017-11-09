import pandas as pd
import h2o
import os
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.automl import H2OAutoML
import time

start = time.time()

data = pd.read_csv("../data/Data_Entry_2017.csv")
bbox = pd.read_csv("../data/BBox_List_2017.csv")


data.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                'Patient_Age', 'Patient_Gender','View_Position',
                'Original_Image_Width', 'Original_Image_Height',
                'Original_Image_Pixel_Spacing_X',
                'Original_Image_Pixel_Spacing_Y', 'Unnamed']


# Drop columns that aren't needed.
data.drop(['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed'], axis = 1, inplace = True)
data.drop(['Original_Image_Width', 'Original_Image_Height'], axis = 1, inplace = True)


# Check if number is in years, months, or days
data['Age_Measure'] = data['Patient_Age'].astype(str).str[3]

# Remove the character in Patient Age, and convert to integer
data['Patient_Age'] = data['Patient_Age'].map(lambda x: str(x)[:-1]).astype(int)

# Remove everything to the right of the '|' delimiter in the labels (preliminary category reduction)
data['Finding_Labels'] = data['Finding_Labels'].apply(lambda x: x.split('|')[0])

# Reduced to 15 total categories
# data.Finding_Labels.value_counts()

# List of sample images (10000 only)
sample_images = os.listdir('../data/images')

h2o.init(nthreads=-1)

data = h2o.H2OFrame(data)

# Drop irrelevant data
data = data.drop('Image_Index')

# Convert features to factors (categorical)
data['Follow_Up_#'] = data['Follow_Up_#'].asfactor()
data['Patient_ID'] = data['Patient_ID'].asfactor()


# Split Data
print("Splitting Data")
split = data.split_frame(ratios=[0.6,0.2], seed=7)

train = split[0]
valid = split[1]
test = split[2]

# Train Model
training_columns = [t for t in train.columns if t != 'Finding_Labels']

# RANDOM FOREST CLASSIFIER
# rf = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
# rf.train(x=training_columns, y='Finding_Labels', training_frame=train)
#
# performance = rf.model_performance(test_data=test)
# print(performance)
# print(rf.r2())

# GBM CLASSIFIER
# print("Training GBM")
# gbm = H2OGradientBoostingEstimator(ntrees=100, distribution='multinomial', max_depth=2, learn_rate=0.1)
# gbm.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid)
# performance = gbm.model_performance(test_data=test)
# # print(performance)
# print("Logloss: ", gbm.logloss())
# print("RMSE: ", gbm.rmse())
# print("R2: ", gbm.r2())




'''Grid Search Hyper Parameters'''
# GBM hyperparameters
# gbm_params1 = {'learn_rate': [0.01, 0.1],
#                 'max_depth': [3, 5, 9],
#                 'sample_rate': [0.8, 1.0],
#                 'col_sample_rate': [0.2, 0.5, 1.0]}



# gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
#                           grid_id='gbm_grid1',
#                           hyper_params=gbm_params1)
# gbm_grid1.train(x=training_columns, y='Finding_Labels',
#                 training_frame=train,
#                 validation_frame=valid,
#                 ntrees=100,
#                 seed=1)
#
#
# gbm_gridperf1 = gbm_grid1.get_grid(sort_by='auc', decreasing=True)
# print(gbm_gridperf1)



aml=H2OAutoML(max_runtime_secs=10)
aml.train(x=training_columns, y='Finding_Labels', training_frame=train, validation_frame=valid, leaderboard_frame=test)

print("Seconds: ", time.time() - start)

if __name__ == '__main__':
    pass
