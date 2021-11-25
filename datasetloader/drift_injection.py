import random
import numpy as np
import pandas as pd

import math


def num_cols(df):
    """

    :param df:
    :return: array of indices corresponding to the numerical columns
    """
    types = [type(df[col].values[1]) for col in df.columns]
    is_num = [not int(x is str) for x in types]
    is_num_idx = np.flatnonzero(is_num)

    return is_num_idx


def create_drift_points(X, min_point=0.7, max_point=0.9):
    """

    :param X: pd.Dataframe of features
    :param min_point: float in [0,1], minimum point of the stream to start drift (default 0.7)
    :param max_point: float in [0,1], maximum point of the stream to start drift (default 0.9)
    :return: dict, "row" drift point, "cols" indices of columns that will be swapped
    """
    driftpoint_perc = np.random.uniform(min_point, max_point, 1)
    driftpoint = int(driftpoint_perc * X.shape[0])
    num_ids = num_cols(X)

    l = len(num_ids) / 2
    l = math.ceil(l)

    ids = random.sample(list(num_ids), l)

    dpoints = dict({"row": driftpoint, "cols": ids})


    return dpoints


def swap_columns(X, y, selected_cols, starting_row, classification=True):
    if classification:
        unique_classes = list(np.unique(y))

    else:
        bins = np.histogram(y.iloc[starting_row:], bins=math.ceil(math.sqrt(len(y.iloc[starting_row:])) / 10))
        good_bins = [i for i in range(len(bins[0])) if bins[0][i] >= 100]
        if len(good_bins) == 0:
            edges = bins[1]
        else:
            edges = bins[1][good_bins]
        edges_pairs = list(zip(edges[::2], edges[1::2]))
        unique_classes = edges_pairs

    selected_class = random.sample(unique_classes, 1)[0]

    df = X.copy()
    df['target'] = y
    df_before = X.iloc[:starting_row, :]
    df = df.iloc[starting_row:, :]

    column_pairs = list(zip(selected_cols[::2], selected_cols[1::2]))

    # swap columns value if the class is the random chosen one (selected class)
    for i in range(len(column_pairs)):
        if classification:
            df[df.columns[list(column_pairs[i])]] = \
                df[df.columns[list(column_pairs[i])]].where(df['target'] !=
                                                            selected_class,
                                                            df[df.columns[list(column_pairs[i])[::-1]]].values)
            df['drifted'] = np.zeros(df.shape[0])
            df['drifted'] = df['drifted'].where(df['target'] != selected_class, 1)
        else:
            df[df.columns[list(column_pairs[i])]] = \
                df[df.columns[list(column_pairs[i])]].where((df['target'] < selected_class[0]) |
                                                            (df['target'] >= selected_class[1]),
                                                            df[df.columns[list(column_pairs[i])[::-1]]].values)
            df['drifted'] = np.zeros(df.shape[0])
            df['drifted'] = df['drifted'].where((df['target'] < selected_class[0]) |
                                                (df['target'] >= selected_class[1]), 1)

    df_before['drifted'] = np.zeros(df_before.shape[0])

    return df_before.append(df.drop(columns=['target']))


def inject_drift(X, y, min_point=0.7, max_point=0.9, classification=True):
    d_point = create_drift_points(X, min_point, max_point)
    X = swap_columns(X, y, d_point['cols'], d_point['row'], classification)

    return X, y, d_point["row"]
