import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# 모델 가져오기
def get_model(path):
    '''
    :param path: 모델 저장되어 있는 경로
    :return: 모델, 모델의 경로
    '''
    model = load_model(path)
    return model, path

# 문장 예측을 위해 이전에 사용한 형태대로 문장 데이터 변환
def convert_sentence(tagger, max_len, input_=True, stop_path=None): # 수정 필요(토크나이저)
    pass
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

# 문장의 라벨 예측
def pred_sentences(sent, model):
    '''
    :param sent: 문장
    :param model: 감성분석 예측에 사용할 모델
    :return: 긍정, 부정, 중립 여부
    '''
    prediction = model.predict(sent)
    label = np.argmax(prediction)
    if label == 0:
        res = "중립"
    elif label == 1:
        res = "긍정"
    else:
        res = "부정"
    return res

# 전체 데이터 예측
def pred_data(model_path, pred_data, save_result=False):
    '''
    :param model_path: 문장 데이터 예측에 사용할 모델 경로
    :param pred_data: 감성분석 예측에 활용할 문장 데이터
    :param save_result: 결과 저장 여부
    :return: 문장 예측 결과를 pandas의 Series 자료형으로 반환
    '''

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