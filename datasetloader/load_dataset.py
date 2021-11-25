import pandas as pd
import numpy as np
from skmultiflow.data.data_stream import DataStream
from sklearn.preprocessing import LabelEncoder
from .drift_injection import inject_drift


def read_data_electricity_market(foldername="data/", shuffle=False):
    df = pd.read_csv(foldername + "elecNormNew.csv")
    if shuffle is True:
        df = df.sample(frac=1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    # Set x,y as numeric
    X = X.astype(float)
    label = ["UP", "DOWN"]
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)

    return X, y


def read_data_weather(foldername="data/weather/", shuffle=False):
    df_labels = pd.read_csv(foldername + "NEweather_class.csv")
    y = df_labels.values.flatten()

    df_data = pd.read_csv(foldername + "NEweather_data.csv")
    df = df_data.copy()
    df['y'] = y

    if shuffle is True:
        df = df.sample(frac=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    return X, y


def read_data_forest_cover_type(foldername="data/", shuffle=False):
    df = pd.read_csv(foldername + "forestCoverType.csv")
    if shuffle is True:
        df = df.sample(frac=1)
    X = df.iloc[:, 1:12]
    y = df.iloc[:, -1:].values.flatten()

    return X, y


def read_data_anas(foldername="data/", shuffle=False):
    df = pd.read_csv(foldername + "panama.csv")
    if shuffle is True:
        df = df.sample(frac=1)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1:]

    return X, y


def load_stream(name, drift=True, shuffle=False):
    """
    Available dataset: 'electricity', 'weather', 'forestcover', 'anas'
    Return a stream of the dataset with injected drift if drift is True, and an array of 1 and 0 corresponding
    to the rows with a drift injected
    :param shuffle: Bool, wheter to shuffle or not the dataset
    :param name: string, dataset to load.
    :param drift: Bool, default is True
    :return: skmultiflow datastream, np.array of drifted rows, int of drift starting point
    """
    if name == 'electricity':
        X, y = read_data_electricity_market(shuffle=shuffle)
        if drift:
            X, y, drift_point = inject_drift(X, y)
            drifted_rows = X['drifted']
    elif name == 'weather':
        X, y = read_data_weather(shuffle=shuffle)
        if drift:
            X, y, drift_point = inject_drift(X, y)
            drifted_rows = X['drifted']
    elif name == 'forestcover':
        X, y = read_data_forest_cover_type(shuffle=shuffle)
        if drift:
            X, y, drift_point = inject_drift(X, y)
            drifted_rows = X['drifted']
    elif name == 'anas':
        X, y = read_data_anas(shuffle=shuffle)
        if drift:
            X, y, drift_point = inject_drift(X, y, classification=False)
            drifted_rows = X['drifted']

    if drift:
        stream = DataStream(X.drop(columns=['drifted']), y)
    else:
        stream = DataStream(X, y)
        drifted_rows = np.zeros(X.shape[0])
        drift_point = np.nan

    return stream, np.array(drifted_rows), drift_point

