from tensorflow.keras.preprocessing.sequence import pad_sequences

# 케라스 토크나이저 이용해 문장 패딩
def pad_sentences(data, max_len=128):
    '''
    :param data: 패딩할 문장 데이터
    :param max_len: 패딩 설정 길이
    :return: 패딩된 문장 데이터
    '''
    data = pad_sequences(data, maxlen=max_len, dtype='long', truncating='post', padding='post')
    return data

