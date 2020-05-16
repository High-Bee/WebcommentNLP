from Data.load_data import prepare_train, prepare_pred
from Data.transform_labels import one_hot
from Data.cleanse_data import get_wordlists
from Models.RNNbased import set_tokenizer, tokenize, check_freq
from Models.RNNbased import check_len, pad_sentences
from Models.RNNbased import create_LSTM
from Models.RNNbased import train_model, plot_history, best_model
from Models.RNNbased import pred_data, get_model
import os
import numpy as np
import pandas as pd

# set path
root_path = "C:/Users/user/PycharmProjects/NLP_SentimentAnalysis"
input_path = f"{root_path}/Input"
output_path = f"{root_path}/Output"
TRAIN_PATH = f"{input_path}/train_input.pickle"
PRED_PATH = f"{input_path}/predict_input.pickle"

# set existing file name
stopwords = "stopwords.txt"

# load data for train
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = prepare_train(TRAIN_PATH, mecab=True)
print(X_train_raw[:10])
print(X_test_raw[:10])

# cleanse data(optional by filter) and get word lists
stopwords_path = os.path.join(root_path, stopwords)
X_train = get_wordlists(X_train_raw, filter=True, file_path=stopwords_path)
X_test = get_wordlists(X_test_raw, filter=True, file_path=stopwords_path)
print(f"train input : {len(X_train)}, test input : {len(X_test)}")

# check frequency
for i in range(1, 10):
    check_freq(i, X_train)

# set Keras tokenizer
_, _, vocab_size = check_freq(3, X_train)
tokenizer_name, tokenizer = set_tokenizer(X_train, vocab_size, oov=False, save_path=root_path)
print(f"{tokenizer_name} : Using only {vocab_size} vocabs in {tokenizer}\n")

# tokenize sentences
X_train, drop_train = tokenize(X_train, tokenizer)
X_test, drop_test = tokenize(X_test, tokenizer)

# check empty sentences(optional)
print(f"Should drop {len(drop_train)} train texts. 10 Samples : {drop_train[:10]}")
print(f"Should drop {len(drop_test)} test texts. 10 Samples : {drop_test[:10]}")
empty_train = X_train_raw[X_train_raw.index.isin(drop_train)]
empty_test = X_test_raw[X_test_raw.index.isin(drop_test)]
print("\nCheck Original Empty Texts\n")
print(f"Sample empty train input \n{empty_train[:10]}\n")
print(f"Sample empty test input \n{empty_test[:10]}\n")

# delete empty sentences(optional)
print("Now Delete Empty Train Texts")
X_train = np.delete(X_train, drop_train)
y_train = y_train_raw.drop(labels = drop_train)
y_train.index = pd.RangeIndex(len(y_train.index))
X_test = np.delete(X_test, drop_test)
y_test = y_test_raw.drop(labels = drop_test)
y_test.index = pd.RangeIndex(len(y_test.index))
print(f"<TRAIN> Input shape : {X_train.shape}, Label length : {len(y_train)}")
print(f"<TEST> Input shape : {X_test.shape}, Label length : {len(y_test)}")

# check sentences' lengths(optional)
# plot_sentences(X_train, save_path=root_path, oov=False)
for i in range(0, 200, 10):
    check_len(i, X_train)

# pad sentences
max_len = 80
X_train = pad_sentences(X_train, max_len)
X_test = pad_sentences(X_test, max_len)

# one hot encode labels
y_train = one_hot(y_train, 3)
y_test = one_hot(y_test, 3)
print(y_train[:10])
print(y_test[:10])

# train with basic lstm model
lstm = create_LSTM(input_num=vocab_size+1, dim_num=100)
history, model_dir = train_model(lstm, "LSTM", inputs=X_train, labels=y_train, save_path=output_path)
plot_history(history)

# test
best_model, best_model_path = get_model(best_model(model_dir, "*.h5"))
print(f"Using {best_model_path}.")
loss_and_metrics = best_model.evaluate(X_test, y_test)
print("test loss: %.4f" % (loss_and_metrics[0]))
print("test accuracy: %.4f" % (loss_and_metrics[1]))

# load data for predict
X_pred_raw = prepare_pred(PRED_PATH, filter=True, mecab=True, verbose=1)
print(X_pred_raw[:10])

# get morphs
stopwords_path = f"{root_path}/stopwords.txt"
X_pred = get_wordlists(X_pred_raw['document'], filter=True, file_path=stopwords_path) # dataframe이므로 'document'로 인덱싱해야 한다.
print(f"prediction input : {len(X_pred)}")
print(X_pred[:10])

# tokenize data
X_pred, drop_pred = tokenize(X_pred, tokenizer)

# check empty data
print(f"Should drop {len(drop_pred)} pred texts. 10 Samples : {drop_pred[:10]}")
empty_pred = X_pred_raw[X_pred_raw.index.isin(drop_pred)]
print(f"Sample empty pred input \n{empty_pred[:10]}\n")

# delete data
print("Now Delete Empty Pred Texts")
X_pred = np.delete(X_pred, drop_pred, axis=0)
print(f"<PRED> Input shape : {X_pred.shape}")

# pad data
X_pred = pad_sentences(X_pred, max_len)

# predict with data
prediction = pred_data(best_model_path, X_pred, save_result=True)
print(prediction[:10])
