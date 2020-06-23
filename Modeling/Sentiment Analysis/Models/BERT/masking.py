# BERT 형식으로 마스킹 진행
def create_masks(data):
    '''
    :param data: 마스킹 진행할 문장 데이터
    :return: 해당 문장 데이터에 대한 마스킹
    '''
    masks = []
    for sent in data:
        mask = [float(s>0) for s in sent]
        masks.append(mask)
    return masks