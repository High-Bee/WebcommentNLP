from Data.load_data import prepare_train
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


# 경로 및 변수 설정
root_path = "C:/Users/user/PycharmProjects/NLP_SentimentAnalysis"
input_path = f"{root_path}/Input"
output_path = f"{root_path}/Output"
TRAIN_PATH = f"{input_path}/train_input.pickle"

stopwords = "stopwords.txt"

# 훈련 데이터 로드
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = prepare_train(TRAIN_PATH, mecab=True)

# 데이터 정제
stopwords_path = os.path.join(root_path, stopwords)
X_train = get_wordlists(X_train_raw, filter=True, file_path=stopwords_path)
X_test = get_wordlists(X_test_raw, filter=True, file_path=stopwords_path)

# 단어 빈도 수 체크
for i in range(1, 10):
    check_freq(i, X_train)

# 케라스 토크나이저 설정 및 저장
_, _, vocab_size = check_freq(3, X_train)
tokenizer_name, tokenizer = set_tokenizer(X_train, vocab_size, oov=False, save_path=root_path)

# 토크나이징
X_train, drop_train = tokenize(X_train, tokenizer)
X_test, drop_test = tokenize(X_test, tokenizer)

# 토크나이징 후 빈 데이터 삭제(optional)
empty_train = X_train_raw[X_train_raw.index.isin(drop_train)]
empty_test = X_test_raw[X_test_raw.index.isin(drop_test)]
X_train = np.delete(X_train, drop_train)
y_train = y_train_raw.drop(labels = drop_train)
y_train.index = pd.RangeIndex(len(y_train.index))
X_test = np.delete(X_test, drop_test)
y_test = y_test_raw.drop(labels = drop_test)
y_test.index = pd.RangeIndex(len(y_test.index))

# 문장 길이 체크
# plot_sentences(X_train, save_path=root_path, oov=False)
for i in range(0, 200, 10):
    check_len(i, X_train)

# 문장 패딩
max_len = 80
X_train = pad_sentences(X_train, max_len)
X_test = pad_sentences(X_test, max_len)

# 라벨 원핫 인코딩
y_train = one_hot(y_train, 3)
y_test = one_hot(y_test, 3)

# LSTM 모델링
lstm = create_LSTM(input_num=vocab_size+1, dim_num=100)
history, model_dir = train_model(lstm, "LSTM", inputs=X_train, labels=y_train, save_path=output_path)
plot_history(history)

# 모델 테스트
best_model, best_model_path = get_model(best_model(model_dir, "*.h5"))
print(f"Using {best_model_path}.")
loss_and_metrics = best_model.evaluate(X_test, y_test)
print("test loss: %.4f" % (loss_and_metrics[0]))
print("test accuracy: %.4f" % (loss_and_metrics[1]))
