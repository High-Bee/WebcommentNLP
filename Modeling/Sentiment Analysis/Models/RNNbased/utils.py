from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from etc.etc import KST_now
import os
from fnmatch import fnmatch
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 모델 훈련
def train_model(architecture, name, inputs, labels, save_path, epochs=20, val_split=0.25):
    '''
    :param architecture: 모델 객체
    :param name: 모델 저장할 때 사용할 이름
    :param inputs: 모델 훈련 시 입력 데이터
    :param labels: 모델 훈련 라벨
    :param save_path: 모델 저장 경로
    :param epochs: 모델 훈련 에폭
    :param val_split: 검증에 사용할 데이터 비율
    :return: 케라스 모델 훈련 history, 모델 저장된 경로
    '''
    model = architecture

    # 모델 저장 경로
    date = KST_now()[4:14]
    model_dir = f"{save_path}/models/{date}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 체크포인트 및 조기 종료 조건 설정
    ck = ModelCheckpoint(filepath="{0}/{1}-checkpoint-{{epoch:02d}}-{{val_loss:.4f}}.h5".format(model_dir, name),
                         monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', verbose=1)

    # 모델 훈련
    print(model.summary())
    history = model.fit(inputs, labels,
                        epochs=epochs,
                        callbacks=[ck,es],
                        validation_split=val_split)

    return history, model_dir

# 모델 히스토리 그림
def plot_history(hist):
    '''
    :param hist: 모델 훈련 시 생성된 히스토리
    :return: 훈련 종료 멘트
    '''

    # plot loss
    fig, loss_ax = plt.subplots()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    plt.show()

    # plot accuracy
    fig, acc_ax = plt.subplots()
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')
    plt.show()

    return print("All Train Done!")

# 최상의 모델 찾기
def best_model(path, pat):
    '''
    :param path: 모델 저장 경로
    :param pat: 모델을 저장한 파일 형식
    :return:
    '''
    pattern = pat
    loss_vals = dict()
    for path, subdirs, files in os.walk(path):
        for file in files:
            if fnmatch(file, pat):
                loss_vals.update({file:int(file[21:25])})
    best_model_path = os.path.join(path, min(loss_vals, key=loss_vals.get))
    print(best_model_path)
    return best_model_path

def plot_model(path):
    pass

# 모델 정보 저장
def save_info(file, model_name, tok_name, max_len):
    '''
    :param file: 모델 생성 시의 정보 기록할 파일 이름
    :param model_name: 모델 훈련 시 저장한 모델 이름
    :param tok_name: 모델 훈련 시 사용한 토크나이저 이름
    :param max_len: 문장 패딩 길이
    '''
    if not os.path.exist(file):
        with open(file, 'w') as f:
            f.write("Model Info\n")
    with open(file, 'w+') as f:
        f.write(f"model: {model_name}, tokenizer: {tok_name}, pad : {max_len}")
