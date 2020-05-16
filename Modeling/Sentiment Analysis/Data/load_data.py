from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def load(path, is_predict):
    if is_predict:
        data = pd.read_pickle(path)[['content', 'content_morph']]
    else:
        data = pd.read_pickle(path)[['content', 'content_morph', 'label']]
    data = data.dropna()
    data.index = pd.RangeIndex(len(data.index))
    return data


def prepare_train(path, mecab, filter=False, test_sets=1, test_split=0.3, seed=42, verbose=0):
    data = load(path, is_predict=False)

    if mecab:
        df = data.drop('content', axis=1)
    elif not mecab:
        df = data.drop('content_morph', axis=1)

    df.columns = ['document', 'label']

    if filter:
        df['document'] = df['document'].str.replace("[^가-힣 ]", "", regex=True)

    splitter = StratifiedShuffleSplit(n_splits=test_sets, test_size=test_split, random_state=seed)
    for train_idx, test_idx in splitter.split(df, df['label']):
        train_data = df.loc[train_idx]
        test_data = df.loc[test_idx]
    train_data.index = pd.RangeIndex(len(train_data.index))
    test_data.index = pd.RangeIndex(len(test_data.index))

    if verbose > 0:
        print(f"train_shape : {train_data.shape}, test_shape: {test_data.shape}")
        print(f"train set label 비율 \n {train_data.label.value_counts() / len(train_data)}")
        print(f"test set label 비율 \n {test_data.label.value_counts() / len(test_data)}")

    X_train = train_data['document'].astype(str)
    y_train = train_data['label'].astype(int)
    X_test = test_data['document'].astype(str)
    y_test = test_data['label'].astype(int)

    return (X_train, y_train), (X_test, y_test)


def prepare_pred(pred_path, mecab, filter=False, verbose=0):
    data = load(pred_path, is_predict=True)
    if mecab:
        df = data.drop('content', axis=1)
    elif not mecab:
        df = data.drop('content_morph', axis=1)

    df.columns = ['document']

    if filter:
        df['document'] = df['document'].str.replace("[^가-힣 ]", "", regex=True)

    if verbose > 0:
        print(f"predict_shape : {df.shape}")
    return df

