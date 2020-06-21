from Data.load_data import prepare_pred
from Data.cleanse_data import get_wordlists
from Models.RNNbased import tokenize
from Models.RNNbased import pad_sentences
from Models.RNNbased import best_model
from Models.RNNbased import pred_data, get_model
import os
import numpy as np

# 경로 및 변수 설정
root_path = "C:/Users/user/PycharmProjects/NLP_SentimentAnalysis"
input_path = f"{root_path}/Input"
PRED_PATH = f"{input_path}/predict_input.pickle"
stopwords = "stopwords.txt"

# 예측 데이터 로드
X_pred_raw = prepare_pred(PRED_PATH, filter=True, mecab=True, verbose=1)

# 형태소 분석
stopwords_path = os.path.join(root_path, stopwords)
X_pred = get_wordlists(X_pred_raw['document'], filter=True, file_path=stopwords_path)

# 토크나이징
tokenizer = 'some tokenizer' # 기록해 둔 결과 바탕으로 토크나이저 로드
X_pred, drop_pred = tokenize(X_pred, tokenizer)

# 빈 데이터 삭제
empty_pred = X_pred_raw[X_pred_raw.index.isin(drop_pred)]
X_pred = np.delete(X_pred, drop_pred, axis=0)

# pad data
max_len = 'some num' # 기록해 둔 결과 바탕으로 문장 길이 선택
X_pred = pad_sentences(X_pred, max_len)

# predict with data
model_dir = 'saved model directory' # 모델 저장한 경로
best_model, best_model_path = get_model(best_model(model_dir, "*.h5"))
prediction = pred_data(best_model_path, X_pred, save_result=True)
