"""
Chase Leibowitz & Jacob Ominsky
ML IS 2023-24 Final Project

Objective: Use LSTM network to predict stock prices using GOOGL stock data

FILE: Run actual stock predictor
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
import joblib

# Load the model and scaler
regressor = load_model('lstm_GOOGL.h5')
scaler = joblib.load('scaler.pkl')

# Load and preprocess the test data
dataset_test = pd.read_csv('GOOGL_test.csv')
actual_stock_price = dataset_test.iloc[:, 1:2].values

dataset_train = pd.read_csv("GOOGL.csv")
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict stock prices
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(actual_stock_price, color='red', label='Actual Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
