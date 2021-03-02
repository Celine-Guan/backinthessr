from gensim.models import Word2Vec
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
from tensorflow.keras.layers import Embedding
from nltk.stem import WordNetLemmatizer
import string
from tensorflow import keras
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling1D

word_to_id = {}
iter_ = 1
for sentence in X_train:
    for word in sentence:
        if word in word_to_id:
            continue
        word_to_id[word] = iter_
        iter_ += 1

vocab_size = len(word_to_id)

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

model.fit(X_train_pad, y_train, validation_split=0.3, epochs=50, batch_size=16, callbacks=[es])

model.evaluate(X_test_pad, y_test)