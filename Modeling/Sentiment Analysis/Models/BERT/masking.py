def create_masks(data):
    masks = []
    for sent in data:
        mask = [float(s>0) for s in sent]
        masks.append(mask)
    return masks