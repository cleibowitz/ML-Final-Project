This repository houses the final project for Chase Leibowitz's and Jacob Ominsky's senior year independent study at Shipley. 
The project uses an LSTM (long-short-term-memory) network to predict stock prices. The model is trained on GOOGL stock data.
The "train" file trains a model using Keras and Tensorflow and saves it.
The "predict" file runs the model to predict GOOGL stock price data and then plots the actual prices vs the predicted prices.

Future plans:
Incorporate API to train on live-updating stock price data
Modify the program so models can be trained on any stock (using API)
Incorporate training bot with predictor for next-day prices (give trading recommendations)
