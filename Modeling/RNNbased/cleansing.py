import re

def get_stopwords(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        corpus = f.read()
    stopwords = corpus.split('\n')
    return stopwords

def get_wordlists(data, filter, **kwargs):
    words = []
    if filter:
        stopwords = get_stopwords(kwargs.get('file_path'))
        for idx in range(len(data)):
            sent = data[idx].split()
            sent = [word for word in sent if not word in stopwords]
            words.append(sent)
    else:
        for idx in data.index:
            sent = data[idx].split()
            sent = [word for word in sent]
            words.append(sent)
    return words