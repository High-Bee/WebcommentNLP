from tensorflow.keras.utils import to_categorical

def one_hot(label, num_classes):
    label = to_categorical(label, num_classes)
    print(f"label shape : {label.shape}")
    return label


# PyTorch : prevent GPU error
def GPU_label(data):
    for i in range(len(data)):
        if data[i] == -1:
            data[i] = 2
    return data