import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Data.load_data import prepare_train
from Data.transform_labels import GPU_label
from Models.BERT.tokenizing import BertTokenize
from Models.BERT.padding import pad_sentences
from Models.BERT.masking import create_masks
from Models.BERT.utils import get_accuracy, format_time
import random
import numpy as np
import os

# 경로 및 변수 설정
root_path = "C:/Users/user/PycharmProjects/NLP_SentimentAnalysis"
input_path = f"{root_path}/Input"
output_path = f"{root_path}/Output"
TRAIN_PATH = f"{input_path}/train_input.pickle"
PRED_PATH = f"{input_path}/predict_input.pickle"
MODEL_PATH = f"{output_path}/models"

BATCH = 32 # 데이터로더 배치
SEED = 42 # 난수 시드

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

# 학습 데이터 로더
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH)

# 검증 데이터 로더
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validataion_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validataion_sampler, batch_size=BATCH)

# 테스트 데이터 로더
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH)

# GPU 장치 검사 및 설정
device_name = tf.test.gpu_device_name()
if device_name == '/devuce:GPU:0':
    print(f"GPU: {device_name}")
    device = torch.device("cuda")
else:
    raise SystemError('GPU 없음.')

# 모델: transformers 라이브러리의 classification 모델.
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3) # 3진 분류
# print(model.cuda()) # 모델 확인

# 옵티마이저: Adam.
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# 학습 횟수 설정
epochs = int(input('학습 횟수 설정: '))
total_steps = len(train_dataloader)*epochs # 총 훈련 스텝

# 스케쥴러
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 시드 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 학습
model.zero_grad() # 그래디언트 초기화
for epoch in range(0, epochs):

    print(f"======== {epoch + 1} / {epochs} ========")

    total_loss = 0 # 한 훈련 에폭 내 loss 초기화
    model.train() # 훈련 모드

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch # 언패킹 통해 배치에서 데이터 추출

        # forward
        outputs = model(b_input_ids,
                        token_type_ids=None, # Q&A 등 형식 아니고, 그냥 분류.
                        attention_masks=b_input_masks,
                        labels=b_labels)

        # 스텝 loss 계산
        loss = outputs[0]
        total_loss += loss # 훈련 에폭 내 loss 업데이트

        # backward
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 가중치 파라미터 업데이트
        optimizer.step()

        # 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 에폭 내 평균 loss 계산
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"           평균 학습 loss: {avg_train_loss}")

    # 에폭 학습 완료 후 검증
    model.eval() # 평가 모드

    eval_acc, nb_eval_steps = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad(): # 평가 수행 시 그래디언트 계산 해제
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # cpu로 로짓, 정확도 계산
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        temp_acc = get_accuracy(logits, label_ids)
        eval_acc += temp_acc
        nb_eval_steps += 1


    # 평균 검증 정확도 계산
    epoch_eval_acc = eval_acc / nb_eval_steps
    print(f"           검증 정확도: {epoch_eval_acc}")


# 학습 완료 후 테스트
model.eval()

test_acc, nb_test_steps = 0, 0

for step, batch in enumerate(test_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    temp_acc = get_accuracy(logits, label_ids)
    test_acc += temp_acc
    nb_test_steps += 1

testSet_accuracy = test_acc / nb_test_steps

print(f" ======== ======== ======== 테스트 정확도: {testSet_accuracy} ======== ======== ======== ")

# 사전훈련 모델 저장
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

pass