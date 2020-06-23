import torch
import numpy as np
from Models.BERT.tokenizing import BertTokenize
from Models.BERT.padding import pad_sentences
from Models.BERT.masking import create_masks

# BERT 모델에 사용하기 위해 문장 변경
def convert_data(sent):
    '''
    :param sent: 감성분석을 진행할 문장
    :return: torch 텐서 형태로 변환된 문장 데이터, 문장 데이터에 맞는 마스킹 데이터
    '''

    # BERT 토크나이저로 토크나이징 후 패딩
    tokenized_sent = BertTokenize(list(sent))
    padded_sent = pad_sentences(tokenized_sent)

    # 어텐션 마스크 생성
    masks_sent = create_masks(padded_sent)

    # 입력 데이터와 마스크를 pytorch 텐서 형태로 변환
    inputs = torch.tensor(padded_sent)
    masks = torch.tensor(masks_sent)

    return inputs, masks

# BERT 모델을 이용해 문장 예측
def test_sentences(sent, model, device):
    '''
    :param sent: 감성분석을 진행할 문장
    :param model: 감성분석에 활용할 BERT 모델
    :param device: 학습에 활용할 GPU 장치 경로
    :return: 문장의 긍정, 부정, 중립 여부
    '''
    # 모델 평가 모드로 전환
    model.eval()

    # 문장 예측
    inputs, masks = convert_data(sent)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # 예측 결과
    logits = outputs[0]
    pred = np.argmax(logits)
    if pred == 1:
        result = "긍정"
    elif pred == 0:
        result = "중립"
    else:
        result = "부정"

    return result