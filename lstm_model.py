import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(50, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Contoh penggunaan:
# model = build_lstm_model((60, 1))
# model.fit(X_train, y_train, epochs=10, batch_size=32)
