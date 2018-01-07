# X-Net: Classifying Chest X-Rays Using Deep Learning

## Background
In October 2017, the National Institute of Health open sourced 112,000+ images of chest
chest x-rays. Now known as ChestXray14, this dataset was opened in order to allow clinicians to make better
diagnostic decisions for patients with various lung diseases.

## Table of Contents
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Pipeline](#pipeline)
5. [Preprocessing](#preprocessing)
6. [Model (Structured Data)](#model-structured-data)
7. [Model (Convolutional Neural Network)](#model-convolutional-neural-network)
8. [Explanations](#explanations)
9. [References](#references)


## Objective
* Train a convolutional neural network to detect and classify diagnoses of patients.
* Couple structured and unstructured datasets together into a dual classifier.


## Dataset
The ChestXray14 [dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
consists of both images and structured data.

The image dataset consists of 112,000+ images, which consist of 30,000 patients.
Some patients have multiple scans, which will be taken into consideration.
All images are originally 1024 x 1024 pixels.

Due to data sourcing & corruption issues, my image dataset consists of 10,000
of the original 112,000 images. All data is used for the structured model.

Additionally, structured data is also given to us for each image. This dataset
includes features such as age, number of follow up visits, AP vs PA scan, and
the patient gender.


## Exploratory Data Analysis

When researching the labels, there are 709 original, unique categories present. On further examination, the labels are hierarchical. For example, some labels are only "Emphysema", while others are "Emphysema | Cardiac Issues".

The average age is 58 years old. However, about 400 patients are labeled
as months, 1 of them is labeled in days.


## Pipeline

Two pipelines were created for each dataset. Each script is labeled as either "Structured" or
"CNN", which indicates which data pipeline the script is part of.

|Description|Script|Model|
| :-------------: |:-------------:|:-------------:|
|EDA|eda.py|Structured
|Resize Images|resize_images.py|CNN
|Reconcile Labels|reconcile_labels.py|CNN
|Convert Images to Arrays|image_to_array.py|CNN
|CNN Model|cnn.py|CNN
|Structured Data Model|model.py|Structured

## Preprocessing

First, the labels were changed to reflect single categories, as opposed to the hierarchical categorical labels in the original
data set. This reduces the number of categories from 709 to 15 categories. The label reduction takes its queue from the Stanford
data scientists, who reduced the labels in the same way.

Irrelevant columns were also removed. These columns either had zero variance, or provided minimal information
on the patient diagnosis.

Finally, anyone whose age was given in months (M) or days (D) was removed. The amount of data removed is minimal,
and does not affect the analysis.


## Model (Structured Data)

The structured data is trained using a gradient boosted classifier. The random
forest classifier was also used. When comparing the results, both were nearly
equal. The GBM classifier was used due to its speed over the random forest,
and due to producing equal or better results to the random forest.


## Results (Structured Data)

|Measurement|Score|
| :-------------: |:-------------:|
|Model | H2O Gradient Boosting Estimator|
|Log Loss|1.670|
|MSE|0.510|
|RMSE|0.714|
|R^2|0.967|
|Mean Per-Class Error|0.933|



## Model (Convolutional Neural Network)

The CNN was trained using Keras, with the TensorFlow backend.

The model is similar to the VGG architectures; 2 to 3 convolution layers are used in each set of layers, followed by a pooling layer.

Dropout is used in the fully connected layers only, which slightly
improved the results.

## Results (Convolutional Neural Network)
|Measurement|Score|
| :-------------: |:-------------:|
|Accuracy|0.5456
|Precision|0.306
|Recall|0.553
|F1|0.394

## Explanations

Per the [blog post](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/) from Luke Oakden-Rayner, there are multiple problems with this dataset. The most notable
are the images (and structured data) being labeled incorrectly. He also notes the annotators did not look at the images.

This became evident when training both models. Despite regularization, and rectifying the class imbalances,
both models learned to return meaningless predictions. Per the above statement, this can be attributed to the incorrect
labeling of the images.


Due to these findings, per Mr. Oakden-Rayner, and my own analysis: "I believe the ChestXray14 dataset, as it exists now, is not fit for training medical AI systems to do diagnostic work."


This doesn't discount convolutional neural networks from being able to predict diseases, but this is dependent on the
labels being correct and accurate. Once this becomes rectified, and the images are correctly labeled, further analysis
can resume against the ChestXray14 dataset.


## Tech Stack

<p align = "center">
<img align="center" src="data/tech_stack.jpg" alt="tech_stack_banner"/>
</p>

## References
[NIH Clinical Center provides one of the largest publicly available chest x-ray datasets to scientific community](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community?utm_content=buffer0bad0&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer)

[Algorithm better at diagnosing pneumonia than radiologists](http://med.stanford.edu/news/all-news/2017/11/algorithm-can-diagnose-pneumonia-better-than-radiologists.html)

[AutoML: Automatic Machine Learning](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

[Stacked Ensembles](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
