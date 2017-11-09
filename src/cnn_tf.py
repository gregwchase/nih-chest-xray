import tensorflow as tf
import numpy as np
import pandas as pd


print("Importing Data")
labels = pd.read_csv("../data/sample_labels.csv")
X = np.load("../data/X_sample.npy")

y = labels['Finding_Labels']
y = np.array(pd.get_dummies(y))

print("Splitting Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

input_layer = tf.reshape(features["x"], [-1, 512, 512, 1])

conv1 = tf.layers.conv2d(inputs=input_layer,
    filters=32,
    kernel_size=[2, 2],
    padding="same",
    activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[2, 2],
    padding="same",
    activation=tf.nn.relu)


pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
logits = tf.layers.dense(inputs=dropout, units=15)
