import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# scikit-multiflow
from skmultiflow.data.data_stream import DataStream

from progress.bar import IncrementalBar


def drift_detector(S, T, threshold):
    """
    Return True if a drift between S and T is detected, False otherwise

    Parameters
    ----------
    S : array-like of shape (n_samples, n_features), Source Data
    T : array-like of shape (n_samples, n_features), Target Data
    threshold: float, threshold for the AUC.

    Returns
    -------
    drift_detected:bool
    """
    T = pd.DataFrame(T)
    S = pd.DataFrame(S)

    # Give slack variable in_target which is 1 for old and 0 for new
    T['in_target'] = 0  # in target set
    S['in_target'] = 1  # in source set

    # Combine source and target with new slack variable
    ST = pd.concat([T, S], ignore_index=True, axis=0)
    labels = ST['in_target'].values
    ST = ST.drop('in_target', axis=1).values

    # You can use any classifier for this step. We advise it to be a simple one as we want to see whether source
    # and target differ not to classify them.
    clf = LogisticRegression(solver='liblinear')
    # clf = svm.SVC(probability = True, class_weight = 'balanced')
    # clf = DecisionTreeClassifier()
    predictions = np.zeros(labels.shape)

    # Divide ST into two equal chunks
    # Train LR on a chunk and classify the other chunk
    # Calculate AUC for original labels (in_target) and predicted ones
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    for train_idx, test_idx in skf.split(ST, labels):
        X_train, X_test = ST[train_idx], ST[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        predictions[test_idx] = clf.predict(X_test)
        # print(probs)
        # predictions[test_idx] = probs
        # print(predictions)
    auc_score = AUC(labels, predictions)

    # Signal drift if AUC is larger than the threshold
    if auc_score > threshold:
        return True
    else:
        return False


class D3():

    def __init__(self, dim, auc=0.7, w=100, rho=0.1):
        """
        Parameters
        ----------
        dim : int
            Number of features
        auc : float
            Threshold for the AUC (default is 0.7)
        w : int
            Size of the old data (default is 100)
        rho : float
            Percentage of new data with respect to old (default is 0.1)
        """
        self.size = int(w * (1 + rho))
        self.win_data = np.zeros((self.size, dim))
        # self.win_label = np.zeros(self.size)
        self.w = w
        self.rho = rho
        self.dim = dim
        self.auc = auc
        self.drift_count = 0
        self.window_index = 0

    def addTrainData(self, X):
        self.win_data[:self.w] = X
        self.window_index = self.w

    def addInstance(self, X):
        if self.isEmpty():
            self.win_data[self.window_index] = X
            # self.win_label[self.window_index] = y
            self.window_index = self.window_index + 1
        else:
            print("Error: Buffer is full!")

    def isEmpty(self):
        return self.window_index < self.size

    def driftCheck(self):
        # Uncomment the four lines to restore original functionality of D3
        if drift_detector(self.win_data[:self.w], self.win_data[self.w:self.size],
                          self.auc):  # returns true if drift is detected

            self.window_index = self.w
            # self.win_data = np.roll(self.win_data, -1*self.w, axis=0)
            # self.win_label = np.roll(self.win_label, -1*self.w, axis=0)
            self.drift_count = self.drift_count + 1

            return True
        else:
            self.window_index = self.w
            # self.win_data = np.roll(self.win_data, -1*(int(self.w*self.rho)), axis=0)
            # self.win_label =np.roll(self.win_label, -1*(int(self.w*self.rho)), axis=0)
            return False

    def getCurrentData(self):
        return self.win_data[:self.window_index]

    def getCurrentLabels(self):
        return self.win_label[:self.window_index]


def d3_inference(train_results, win_lenght=100, rho=0.1, auc_score=0.7):
    n_train = train_results["n_train"]
    stream = train_results["Stream"]
    X_train = train_results["X_train"]

    stream.restart()
    stream.next_sample(n_train)

    D3_win = D3(stream.data.shape[1], w=win_lenght, rho=rho, auc=auc_score)

    D3_win.addTrainData(X_train[-win_lenght:])

    results = {'detected_drift_points': []}

    bar = IncrementalBar('D3_inference', max=stream.n_remaining_samples())
    i = n_train
    while stream.has_more_samples():
        bar.next()
        X = stream.next_sample()[0]
        if D3_win.isEmpty():
            D3_win.addInstance(X)
        else:
            if D3_win.driftCheck():  # detected
                results['detected_drift_points'].append(i)
            else:
                pass
        i += 1
    bar.finish()

    return results