import numpy as np
import datetime

# 정확도 계산
def get_accuracy(preds, labels):
    '''
    :param preds: 모델로 예측한 라벨
    :param labels: 실제 라벨
    :return: 예측 라벨과 실제 라벨의 일치 여부 바탕으로 계산한 단순 정확도
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)/len(labels_flat)

# 시간 표시 형식 변환
def format_time(elapsed):
    '''
    :param elapsed: 소요 시간 
    :return:변환된 시간 형태
    '''
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))
