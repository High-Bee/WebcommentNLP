from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from etc.etc import KST_now
import os
from fnmatch import fnmatch
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def train_model(architecture, name, inputs, labels, save_path, epochs=20, val_split=0.25):
    model = architecture

    # directory model will be saved
    date = KST_now()[4:14]
    model_dir = f"{save_path}/models/{date}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # set checkpoint and early stopping options
    ck = ModelCheckpoint(filepath="{0}/{1}-checkpoint-{{epoch:02d}}-{{val_loss:.4f}}.h5".format(model_dir, name),
                         monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', verbose=1)

    # train
    print(model.summary())
    history = model.fit(inputs, labels,
                        epochs=epochs,
                        callbacks=[ck,es],
                        validation_split=val_split)

    return history, model_dir

def plot_history(hist):

    # plot loss
    fig, loss_ax = plt.subplots()
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')
    plt.show()

    # plot accuracy
    fig, acc_ax = plt.subplots()
    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')
    plt.show()

    return print("All Train Done!")

def best_model(path, pat):
    pattern = pat # saved model format
    loss_vals = dict()
    for path, subdirs, files in os.walk(path):
        for file in files:
            if fnmatch(file, pat):
                loss_vals.update({file:int(file[21:25])})
    best_model_path = os.path.join(path, min(loss_vals, key=loss_vals.get))
    print(best_model_path)
    return best_model_path

def plot_model(path):
    pass

def save_info(file, model_name, tok_name, max_len):
    if not os.path.exist(file):
        with open(file, 'w') as f:
            f.write("Model Info\n")
    with open(file, 'w+') as f:
        f.write(f"model: {model_name}, tokenizer: {tok_name}, pad : {max_len}")
