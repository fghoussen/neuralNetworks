# lstm for time series forecasting
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR) # turn off tensor flow messages
tf.autograph.set_verbosity(3) # turn off tensor flow messages
from numpy import sqrt, asarray
from pandas import read_csv
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)
# load the dataset
df = read_csv('monthly-car-sales.csv', header=0, index_col=0, squeeze=True)
# retrieve the values
values = df.values.astype('float32')
# specify the window size
n_steps = 5
# split into samples
X, y = split_sequence(values, n_steps)
# reshape into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# split into train/test
n_test = 12 # Use last 12 months as testing set.
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]
print('Data shape: train', X_train.shape, y_train.shape, 'test', X_test.shape, y_test.shape)
# define model
x_inp = Input(shape=(n_steps, 1))
x_tmp = LSTM(100, activation='relu', kernel_initializer='he_normal')(x_inp)
x_tmp = Dense(50, activation='relu', kernel_initializer='he_normal')(x_tmp)
x_tmp = Dense(50, activation='relu', kernel_initializer='he_normal')(x_tmp)
x_out = Dense(1)(x_tmp)
model = Model(inputs=x_inp, outputs=x_out)
model.summary()
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=100)
# fit the model
history = model.fit(X_train, y_train, epochs=350, batch_size=32, verbose=0, validation_data=(X_test, y_test), validation_split=0.3, callbacks=[es])
# save model to file
model.save('model.h5')
# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('MSE')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()
# evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('Test set - mse: %.3f, rmse: %.3f, mae: %.3f' % (mse, sqrt(mse), mae))
# load the model from file
model = load_model('model.h5')
# make a prediction
row, target = [float(val) for val in X_test[0]], y_test[0]
yhat = model.predict([row])
print('Predicted: %.3f, target: %.3f' % (yhat, target))
