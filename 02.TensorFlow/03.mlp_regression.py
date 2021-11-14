# mlp for regression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # turn off tensor flow messages
tf.autograph.set_verbosity(3) # turn off tensor flow messages
from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from keras import backend as K
def r2_score(y_true, y_pred):
    SSR = K.sum(K.square(y_true - y_pred))
    SST = K.sum(K.square(y_true - K.mean(y_true)))
    return (1. - SSR/(SST + K.epsilon()))
# load the dataset
df = read_csv('housing.csv', header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print('data shape: train', X_train.shape, y_train.shape, 'test', X_test.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
x_inp = Input(shape=(n_features,))
x_tmp = Dense(10, activation='relu', kernel_initializer='he_normal')(x_inp)
x_tmp = Dense(8, activation='relu', kernel_initializer='he_normal')(x_tmp)
x_out = Dense(1, activation='linear')(x_tmp)
model = Model(inputs=x_inp, outputs=x_out)
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=[r2_score])
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# evaluate the model
error, r2_score = model.evaluate(X_test, y_test, verbose=0)
print('test set - mse: %.3f, rmse: %.3f, r2_score: %.3f ' % (error, sqrt(error), r2_score))
# make a prediction
row, target = [float(val) for val in X_test[0]], y_test[0]
yhat = model.predict([row])
print('predicted: %.3f, target: %.3f' % (yhat, target))
