def pad_sentences(data, max_len=128):
    data = pad_sequences(data, maxlen=max_len, dtype='long', truncating='post', padding='post')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
