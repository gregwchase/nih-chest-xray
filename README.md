# X-Net: Classifying Chest X-Rays Using Deep Learning

## Background
In October 2017, the National Institute of Health open sourced 100,000 images of chest
chest x-rays. The datasets was opened in order to allow clinicians to make better
diagnostic decisions for patients.


# Dataset
The [dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
consists of both images and structured data.

The image dataset consists of 112,000 images, which consist of 30,000 patients.
Some patients have multiple scans, which will be taken into consideration.
All images are originally 1024 x 1024 pixels.


Additionally, structured data is also given to us for each image. This dataset
includes features such as age, number of follow up visits, AP vs PA scan, and
the patient gender.


## Objective
* Train a convolutional neural network to detect and classify diagnoses of patients.

* Couple structured and unstructured datasets together into a dual classifier.


## Pipeline

* eda.py
* resize_images.py
* reconcile_labels.py
* image_to_array.py
* cnn.py

## Exploratory Data Analysis

When researching the labels, there are 709 original, unique categories present. On further examination, the labels are hierarchical. For example, some labels are only "Emphysema", while others are "Emphysema | Cardiac Issues".

The average age is 58 years old. However, about 400 patients are labeled
as months, 1 of them is labeled in days.


## Preprocessing

First, the labels were changed to reflect single categories, as opposed to the hierarchical categorical labels in the original
data set. This reduces the number of categories from 709 to 15 categories. The label reduction takes its queue from the Stanford
data scientists, who reduced the labels in the same way.

Irrelevant columns were also removed. These columns either had zero variance, or provided minimal information
on the patient diagnosis.

Finally, anyone whose age was given in months (M) or days (D) was removed. The amount of data removed is minimal,
and does not affect the analysis.

## Model (Structured Data)

The structured data is trained using a distributed random forest. This exists
within the H2O.ai framework.

## Results (Structured Data)

|Measurement|Score|
| :-------------: |:-------------:|
Log Loss|1.400|
MSE|0.450|
RMSE|0.670|
R^2|0.967|
Mean Per-Class Error|0.933|



## Model (Convolutional Neural Network)


## Results (Convolutional Neural Network)
|Measurement|Score|
| ------------- |:-------------:|


## Tech Stack

* H2O.ai
* Keras
* MXNet
* NVIDIA NGC
* Pandas


## References
[NIH Clinical Center provides one of the largest publicly available chest x-ray datasets to scientific community](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community?utm_content=buffer0bad0&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer)

[Algorithm better at diagnosing pneumonia than radiologists](http://med.stanford.edu/news/all-news/2017/11/algorithm-can-diagnose-pneumonia-better-than-radiologists.html)

[AutoML: Automatic Machine Learning](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

[Stacked Ensembles](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
