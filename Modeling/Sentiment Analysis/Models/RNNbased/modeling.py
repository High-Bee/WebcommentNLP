import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Masking, Bidirectional, Dense, Embedding, SimpleRNN, LSTM, GRU, Dropout

def create_LSTM(input_num, dim_num):
    model = Sequential()
    model.add(Embedding(input_num, dim_num, mask_zero=True))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_GRU(input_num, dim_num):
    model = Sequential()
    model.add(Embedding(input_num, dim_num, mask_zero=True))
    model.add(GRU(128))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model