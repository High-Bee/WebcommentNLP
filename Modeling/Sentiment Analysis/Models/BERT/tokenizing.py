from transformers import BertTokenizer

def BertTokenize(data, pretrained):
    tokenizer = BertTokenizer.from_pretrained(pretrained, do_lower_case=False)

    # add special tokens
    sentences = ["[CLS] " + str(sent) + " [SEP]" for sent in data]
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    result = []

    # encode into integers
    for text in tokenized_texts:
        encoded_sent = tokenizer.convert_tokens_to_ids(text)
        result.append(encoded_sent)

    return result