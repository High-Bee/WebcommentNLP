import torch
import numpy as np
from Models.BERT.tokenizing import BertTokenize
from Models.BERT.padding import pad_sentences
from Models.BERT.masking import create_masks

def convert_data(sent):
    # tokenize and pad sentence
    tokenized_sent = BertTokenize(list(sent))
    padded_sent = pad_sentences(tokenized_sent)

    # create attention mask
    masks_sent = create_masks(padded_sent)

    # torch tensor
    inputs = torch.tensor(padded_sent)
    masks = torch.tensor(masks_sent)

    return inputs, masks

def test_sentences(sent, model, device):
    # model to evaluation mode
    model.eval()

    # predict data
    inputs, masks = convert_data(sent)
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # get prediction
    logits = outputs[0]
    pred = np.argmax(logits)
    if pred == 1:
        result = "긍정"
    elif pred == 0:
        result = "중립"
    else:
        result = "부정"

    return result