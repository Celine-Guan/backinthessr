import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
from tensorflow.keras.layers import Embedding
from tensorflow import keras
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D

def train_model(X_train_pad, y_train, vocab_size):
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size +1, output_dim=10, mask_zero=True, input_length=150))
    model.add(SpatialDropout1D(0.2))
    model.add(layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'))
    model.add(SpatialDropout1D(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(loss='mse',
                optimizer='rmsprop', 
                metrics=['mae'])
    model.fit(X_train_pad, y_train, validation_split=0.3, epochs=1, batch_size=16, callbacks=[es])
    return model

def evaluate(model, X_test_pad, y_test):
    return model.evaluate(X_test_pad, y_test)