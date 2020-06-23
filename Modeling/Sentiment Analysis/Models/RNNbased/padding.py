import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from etc.etc import KST_now
import os

# 문장 길이 구성을 그림으로 나타내고 저장
def plot_sentences(data, save_path, oov):
    '''
    :param data: 토크나이징 후 문장 데이터
    :param save_path: 그림을 저장할 경로
    :param oov: Out Of Vocabulary 지정 여부
    '''
    print(f"Max Length : {max(len(sent) for sent in data)}")
    print(f"Average Length : {sum(map(len, data))/len(data)}")
    plt.hist([len(sent) for sent in data], bins=50)
    plt.xlabel("len of sents")
    plt.ylabel("num of sents")
    plt.figure(figsize=(15, 8))
    sent_dir = os.path.join(save_path, 'Length_of_Sentences')
    sent_pic = f"{sent_dir}_oov {str(oov)}.png"
    if not os.path.exists(sent_pic):
        plt.savefig(sent_pic)
    plt.show()

# 문장 길이 체크
def check_len(thre, data):
    '''
    :param thre: 해당 길이 이하의 문장 데이터 빈도 체크
    :param data: 토크나이징된 문장 데이터
    '''
    cnt = 0
    for sent in data:
        if len(sent) <= thre:
            cnt += 1
    print(f"Sentences under {thre} : {(cnt / len(data))*100}%")

# 케라스 패딩을 이용해 문장 패딩
def pad_sentences(data, maxlen):
    '''
    :param data: 문장 데이터
    :param maxlen: 패딩을 진행할 최대 길이
    :return: 패딩된 문장 데이터
    '''
    data = pad_sequences(data, maxlen=maxlen) # 이후 truncating, padding 방식 변경 가능.
    print(data[:3]) # 체크용
    return data