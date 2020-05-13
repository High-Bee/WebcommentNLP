import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from RNNbased.cleansing import get_stopwords, get_wordlists
from RNNbased.padding import pad_sentences
import pickle
import pandas as pd

def get_model(path):
    model = load_model(path)
    return model, path

# def convert_sentence(tagger, max_len, input_=True, stop_path=None): 수정 필요(토크나이저)
#     if not input_:
#         sent = input_
#     sent = input()
#     text = tagger.morphs(sent)
#     if stop_path is not None:
#         stopwords = get_stopwords(stop_path)
#         text = [[word for word in text if not word in stopwords]]
#     print(text)
#     tokens = Tokenizer().texts_to_sequences(text) # using Keras Tokenizer
#     padded_text = pad_sentences(tokens, maxlen=max_len)
#     return padded_text

def pred_sentences(sent, model):
    prediction = model.predict(sent)
    label = np.argmax(prediction)
    if label == 0: res = "unknown"
    elif label == 1: res = "pos"
    else: res = "neg"
    return res

def pred_data(model_path, pred_data, save_result=False):

    model = load_model(model_path)

    pred_raw = model.predict(pred_data)
    print(f"predict data shape : {pred_data.shape}")
    result = np.argmax(pred_raw, axis=1).flatten()
    print(f"length of prediction : {len(result)}")
    preds = pd.Series(result)

    if save_result:
        global output_path
        backslash = '\\'
        pred_data = pd.concat([pd.Series(pred_data), preds], axis=1)
        pred_data.columns = ['document', 'label']
        result.to_csv(f"{output_path}/{model_path.split(backslash)[-1]}.csv", index= False, encoding= "utf-8-sig")

    return result