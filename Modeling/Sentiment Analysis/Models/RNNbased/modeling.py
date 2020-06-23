import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Masking, Bidirectional, Dense, Embedding, SimpleRNN, LSTM, GRU, Dropout

# 기본 LSTM 모델 생성
def create_LSTM(input_num, dim_num):
    '''
    :param input_num: vocab size + 1
    :param dim_num: 임베딩 차원 수
    :return: LSTM 모델
    '''
    model = Sequential()
    model.add(Embedding(input_num, dim_num, mask_zero=True))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 기본 GRU 모델 생성
def create_GRU(input_num, dim_num):
    '''
    :param input_num:  vocab size + 1
    :param dim_num: 임베딩 차원 수
    :return: GRU 모델
    '''
    model = Sequential()
    model.add(Embedding(input_num, dim_num, mask_zero=True))
    model.add(GRU(128))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model