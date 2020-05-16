import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from etc.etc import KST_now
import os

def plot_sentences(data, save_path, oov):
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

def check_len(thre, data):
    cnt = 0
    for sent in data:
        if len(sent) <= thre:
            cnt += 1
    print(f"Sentences under {thre} : {(cnt / len(data))*100}%")

def pad_sentences(data, maxlen):
    data = pad_sequences(data, maxlen=maxlen)
    print(data[:3])
    return data