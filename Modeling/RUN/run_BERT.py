from Data.load_data import prepare_train
from Data.transform_labels import GPU_label
from Models.BERT.tokenizing import BertTokenize

# 경로 설정
root_path = "C:/Users/user/PycharmProjects/NLP_SentimentAnalysis"
input_path = f"{root_path}/Input"
output_path = f"{root_path}/Output"
TRAIN_PATH = f"{input_path}/train_input.pickle"
PRED_PATH = f"{input_path}/predict_input.pickle"

# 훈련 데이터 로드
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = prepare_train(TRAIN_PATH, mecab=False)

# 음수 라벨 바꾸기
y_train = GPU_label(y_train_raw)
y_test = GPU_label(y_test_raw)

# BertTokenizer 이용하여 토크나이징
train_input = BertTokenize(X_train_raw, 'bert-base-multilingual-cased')
test_input = BertTokenize(X_train_raw, 'bert-base-multilingual-cased')
