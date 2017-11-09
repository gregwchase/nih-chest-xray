import numpy as np
import mxnet as mx
import pandas as pd
import os
from sklearn.model_selection import train_test_split


batch_size = 1000
nb_classes = 2
nb_epoch = 20

# Specify parameters before model is run.
img_rows, img_cols = 512, 512
channels = 3
nb_filters = 32
# pool_size = (2,2)
kernel_size = (16,16)

print("Importing Data")
labels = pd.read_csv("../data/sample_labels.csv")
X = np.load("../data/X_sample.npy")

y = labels['Finding_Labels']
y = np.array(pd.get_dummies(y))

print("Splitting Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)


print("Creating Model")
data = mx.symbol.Variable('data')

# First conv layer
conv1 = mx.symbol.Convolution(data=data, kernel=(2,2), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",kernel=(2,2), stride=(2,2))

# Second conv layer
conv2 = mx.symbol.Convolution(data=pool1, kernel=(2,2), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))

# First fully connected
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=15)
# loss
lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

device = mx.cpu()

print("Training")
# create a trainable module on CPU
lenet_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
# train with the same
# lenet_model.fit(X_train, y_train,
#                 # eval_data=val_iter,
#                 optimizer='sgd',
#                 optimizer_params={'learning_rate':0.1},
#                 eval_metric='acc',
#                 # batch_end_callback = mx.callback.Speedometer(batch_size, 100),
#                 num_epoch=2)
