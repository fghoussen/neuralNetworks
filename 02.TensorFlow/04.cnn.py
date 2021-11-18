# example of a cnn for image classification
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # turn off tensor flow messages
tf.autograph.set_verbosity(3) # turn off tensor flow messages
from numpy import asarray, unique, argmax, set_printoptions
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# load dataset
(X_train, y_train), (X_test, y_test) = load_data()
# reshape data to have a single channel
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
print('Data shape: train', X_train.shape, y_train.shape, 'test', X_test.shape, y_test.shape)
# determine the shape of the input images
inp_shape = X_train.shape[1:]
# determine the number of classes
n_classes = len(unique(y_train))
print('Image size:', inp_shape, 'n_classes:', n_classes)
# normalize pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
# define model
x_inp = Input(shape=inp_shape)
x_tmp = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', input_shape=inp_shape)(x_inp)
x_tmp = MaxPool2D((2, 2))(x_tmp)
x_tmp = Flatten()(x_tmp)
x_tmp = Dense(100, activation='relu', kernel_initializer='he_uniform')(x_tmp)
x_tmp = Dropout(0.5)(x_tmp)
x_out = Dense(n_classes, activation='softmax')(x_tmp)
model = Model(inputs=x_inp, outputs=x_out)
model.summary()
# define loss and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test set - accuracy: %.3f, loss: %.3f' % (acc, loss))
# make a prediction
image, target = X_test[0], y_test[0]
yhat = model.predict(asarray([image]))
set_printoptions(formatter={'float': '{: 0.3f}'.format})
print('Predicted: %s (class=%d), target: %s' % (yhat, argmax(yhat), target))
