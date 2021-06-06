import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os
import numpy as np

# load data
X = pickle.load(open('pickle_files/X_train.pickle', 'rb'))
y = pickle.load(open('pickle_files/y_train.pickle', 'rb'))
X = X/255.
y = np.array(y)

timestamp = int(time.time())

# build out the model
conv_layers = 3
dense_layers = 0

NAME = 'c{}d{}'.format(conv_layers, dense_layers)
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))


checkpoint_cb = keras.callbacks.ModelCheckpoint('models/{}.h5'.format(NAME), save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights = True)

model = Sequential()
# first convolution
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) #window of 3x3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# additional convolutions
for i in range(conv_layers - 1):
    model.add(Conv2D(64, (3,3))) #window of 3x3
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# dense layers
for i in range(dense_layers):
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer= 'adam',
            metrics = ['accuracy'])
model.fit(X, y, batch_size=32, epochs = 2, validation_split=0.1,
          callbacks=[checkpoint_cb, early_stopping_cb])
