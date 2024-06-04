"""
Chase Leibowitz & Jacob Ominsky
ML IS 2023-24 Final Project

Objective: Use LSTM network to predict stock prices using GOOGL stock data

FILE: Train and save model
"""

# Import libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime

# Load and prepare data
dataset_train = pd.read_csv("GOOGL.csv")
training_set = dataset_train.iloc[:, 1:2].values

# Normalize training set
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = scaler.fit_transform(training_set)

# Creating X_train and Y_train data structures
X_train = []
Y_train = []
for i in range(60, 1258):
    X_train.append(scaled_training_set[i-60:i, 0])
    Y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

# Directory where the logs will be stored
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='batch')

# Compile and train the model
model.compile(optimizer='nadam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=20, batch_size=64, callbacks=[tensorboard_callback])

# Save the model and scaler
model.save('lstm_GOOGL.h5')
joblib.dump(scaler, 'scaler.pkl')
