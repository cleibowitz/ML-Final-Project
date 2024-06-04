"""
Chase Leibowitz & Jacob Ominsky
ML IS 2023-24 Final Project

Objective: Use LSTM network to predict stock prices using GOOGL stock data

FILE: Train and save model
"""

# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import tensorflow
import keras
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional, BatchNormalization
import joblib
from keras.layers import Flatten
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import datetime

# initialize full dataset
dataset_train = pd.read_csv("GOOGL.csv")

# pick training set
training_set = dataset_train.iloc[:,1:2].values

# normalize training set
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

# creating X_train and Y_train data structures
X_train = []
Y_train = []
for i in range (60, 1258):
    X_train.append(scaled_training_set[i-60:i, 0])
    Y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# build the model by adding the layers to the LSTM
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(Flatten())


# Dense layers with dropout
regressor.add(Dense(units=64, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units=32, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units=1))


# Directory where the logs will be stored
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

# fitting the model

regressor.compile(optimizer = Nadam(), loss = 'mean_squared_error')
regressor.fit(X_train, Y_train, epochs=20, batch_size=64, callbacks=[tensorboard_callback])

regressor.save('lstm_GOOGL.h5')

joblib.dump(scaler, 'scaler.pkl')
