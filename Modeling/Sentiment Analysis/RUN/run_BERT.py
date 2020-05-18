import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from Data.load_data import prepare_train
from Data.transform_labels import GPU_label
from Models.BERT.tokenizing import BertTokenize
from Models.BERT.padding import pad_sentences
from Models.BERT.masking import create_masks
from Models.BERT.utils import get_accuracy, format_time
import random
import numpy as np

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

# 문장 길이 제한 및 패딩
train_input_ids = pad_sentences(train_input)
test_input_ids = pad_sentences(test_input)

# 어텐션 마스크 생성
train_input_masks = create_masks(train_input_ids)
test_input_masks = create_masks(test_input_ids)

# 훈련 세트의 입력, 라벨 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_input_ids, y_train, random_state=42, test_size=0.25)

# 마스크 분리
train_masks, validation_masks, _, _ = train_test_split(train_input_masks, train_input_ids, random_state=42, test_size=0.25)

# 훈련 데이터 텐서 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

# 테스트 데이터 텐서 변환
test_inputs = torch.tensor(test_input_ids)
test_labels = torch.tensor(y_test)
test_masks = torch.tensor(test_input_masks)

# 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)