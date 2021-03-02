import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer
import string
import os
import nltk
nltk.download('wordnet')

def get_data():
    data_path = os.path.dirname(os.path.dirname(__file__))
    X = pd.read_csv(os.path.join(data_path, 'raw_data', 'clean_sentiment_data', 'X_train'))
    y = pd.read_csv(os.path.join(data_path, 'raw_data', 'clean_sentiment_data', 'y_train'))
    X = X['Phrase']
    y = y['Answer.sentiment']
    return X, y

def remove_punctuation(text):
    for punctuation in string.punctuation: 
        text = text.replace(punctuation, ' ') 
    return text

def lowercase (text): 
    lowercased = text.lower() 
    return lowercased

def lemma(text):
    lemmatizer = WordNetLemmatizer() # Initiate lemmatizer
    lemmatized = lemmatizer.lemmatize(text)
    #lemmatized_string = " ".join()
    return lemmatized

def split_sentences(X):
    return [sentence.split(' ') for sentence in X]

def tokenize(sentences, word_to_id):
    return [[word_to_id[_] for _ in s if _ in word_to_id] for s in sentences]

def clean_data(X):
    X = X.apply(remove_punctuation)
    X = X.apply(lowercase)
    X = X.apply(lemma)
    arr = X.to_numpy()
    X = split_sentences(arr)
    return X
    
class Vocabulary():

    def __init__(self, X):
        self.X = X

    def fit(self):
        word_to_id = {}
        iter_ = 1
        for sentence in X:
            for word in sentence:
                if word in word_to_id:
                    continue
                word_to_id[word] = iter_
                iter_ += 1
        vocab_size = len(word_to_id)
        return word_to_id, vocab_size

    def transform(self):
        X_token_train = tokenize(X_train, word_to_id)
        X_train_pad = pad_sequences(X_token_train, dtype='float32', padding='post', value=0, maxlen=150)