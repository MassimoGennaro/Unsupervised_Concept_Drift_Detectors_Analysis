import pandas as pd
import numpy as np
from math import sqrt, log
from scipy.stats import t
from skmultiflow.data.sea_generator import SEAGenerator
from sklearn.naive_bayes import GaussianNB
import random
import warnings
from progress.bar import IncrementalBar
import math
import matplotlib.pyplot as plt
import seaborn as sns
from skmultiflow.data import DataStream

warnings.filterwarnings("ignore")


class Distance:
    """
    Class to compute Hellinger Distance and Jensen-Shannon Divergence
    """

    def hellinger_dist(self, P, Q):
        """
        P : Dictionary containing proportions for each category in one window
        Q : Dictionary for the next window
        """
        diff = 0
        for key in P.keys():
            diff += (sqrt(P[key]) - sqrt(Q[key])) ** 2
        return 1 / sqrt(2) * sqrt(diff)

    def KL_divergence(self, P, Q):
        """
        This method is used in Jensen_Shannon_divergence
        """
        div = 0
        for key in list(P.keys()):
            if P[key] != 0:  # Otherwise P[key]*logP[key]=0
                div += P[key] * log(P[key] / Q[key])

        return div

    def Jensen_Shannon_divergence(self, P, Q):
        """
        P : Dictionary containing proportions for each category in one window
        Q : Dictionary for the next window
        """
        M = {}
        for key in list(P.keys()):
            M.update({key: (P[key] + Q[key]) / 2})

        return 1 / 2 * (self.KL_divergence(P, M) + self.KL_divergence(Q, M))


def discretizer(data, n_bins, method):
    """
    Parameters
    ----------
    data : np.array of numerical values to be discretized
    n_bins: int, how many bins create
    method : str, either 'equalsize' or 'equalquantile'

    Return
    ------
    discretized array : np.array discretized in n_bins with
                        selected method
    """

    if method == 'equalsize':
        return pd.cut(data, n_bins)
    if method == 'equalquantile':
        return pd.qcut(data, n_bins)


def process_bin(data, numerical_cols):
    """
    Transforms values (intervals) of numerical_cols to integers because
    classifiers have problems to deal with names that contain [ , ] , <
    """

    for feature in numerical_cols:
        data[feature] = data[feature].apply(str)
        n = len(data[feature].unique())
        dic = {}
        for i in range(n):
            dic.update({data[feature].unique()[i]: i})
        data[feature] = data[feature].map(dic)
    return data


def generate_proper_dic(window, union_values):
    """
    union_values : list containing union of unique values of the two windows
    """
    dic = {}
    df = window.value_counts()
    n = window.shape[0]
    for key in union_values:
        if key in window.unique():
            dic.update({key: df.loc[key] / n})
        else:
            dic.update({key: 0})
    return dic


class HDDDM():
    def __init__(self, data, gamma=1., alpha=None, distance='Hellinger'):
        if gamma is None and alpha is None:
            raise ValueError("Gamma and alpha can not be None at the same time! "
                             "Please specify either gamma or alpha")
        self.gamma = gamma
        self.alpha = alpha
        self.n_bins = int(np.floor(np.sqrt(data.shape[0])))
        if distance == 'Hellinger':
            self.distance = Distance().hellinger_dist

        # Initialization
        self.baseline = self.add_data(data)
        self.t_denom = 0
        self.n_samples = data.shape[0]
        self.old_dist = 0.
        self.epsilons = []
        self.betas = []

    def add_data(self, data):
        X = data.copy()
        X_cat = X.select_dtypes(include='category')
        X_num = X.select_dtypes(include='number')

        data_tmp = pd.DataFrame()

        for c in X_cat.columns:
            data_tmp[c] = X_cat[c]

        # Discretizing X_num
        for c in X_num.columns:
            data_tmp[c] = discretizer(X_num[c], self.n_bins, 'equalsize')
            data_tmp[c] = data_tmp[c].astype('category')

        data_final = process_bin(data_tmp, X_num.columns)
        return data_final

    def windows_distance(self, ref_window, current_window):

        actual_dist = 0

        for feature in self.baseline.columns:
            ref_liste_values = ref_window[feature].unique()
            current_liste_values = current_window[feature].unique()
            union_values = list(set(ref_liste_values) | set(current_liste_values))
            ref_dic = generate_proper_dic(ref_window[feature], union_values)

            current_dic = generate_proper_dic(current_window[feature], union_values)

            actual_dist += self.distance(ref_dic, current_dic)

        actual_dist /= len(self.baseline.columns)

        return actual_dist

    def add_new_batch(self, X):
        if int(np.floor(np.sqrt(X.shape[0]))) != self.n_bins:
            raise ValueError('Size of new batch must be equal to the size of baseline data')

        self.drift_detected = False
        self.t_denom += 1
        self.curr_batch = self.add_data(X)

        curr_dist = self.windows_distance(self.baseline, self.curr_batch)

        n_samples = X.shape[0]

        eps = curr_dist - self.old_dist
        self.epsilons.append(eps)

        epsilon_hat = (1. / (self.t_denom)) * np.sum(np.abs(self.epsilons))
        sigma_hat = np.sqrt(np.sum(np.square(np.abs(self.epsilons) - epsilon_hat)) / (self.t_denom))

        beta = 0.
        if self.gamma is not None:
            beta = epsilon_hat + self.gamma * sigma_hat
        else:
            beta = epsilon_hat + t.ppf(1.0 - self.alpha / 2, self.n_samples + n_samples - 2) * sigma_hat / np.sqrt(
                self.t_denom)
        self.betas.append(beta)
        # Test for drift
        drift = np.abs(eps) > beta

        if drift == True:
            self.drift_detected = True
            # Uncomment the following lines to restore the original functionality of HDDDM
            # self.baseline = self.add_data(X)
            # self.t_denom = 0
            # self.n_samples = X.shape[0]
            # self.old_dist = 0.
            # self.epsilons = []
            # self.betas = []
        else:
            self.n_samples += n_samples
            self.baseline = pd.concat((self.baseline,
                                       self.add_data(X)))


def hdddm_inference(train_results, win_lenght=1000, gamma=1., alpha=None):
    n_train = train_results["n_train"]
    stream = train_results["Stream"]
    X_train = train_results["X_train"]

    stream.restart()
    stream.next_sample(n_train)

    train_df = pd.DataFrame(X_train[-win_lenght:])

    hdddm = HDDDM(train_df, gamma=gamma, alpha=alpha)

    results = {'detected_drift_points': []}
    i = n_train
    bar = IncrementalBar('HDDDM_inference', max=math.ceil(stream.n_remaining_samples() / win_lenght))
    while stream.has_more_samples():
        bar.next()
        X = stream.next_sample(win_lenght)[0]
        X_df = pd.DataFrame(X, columns=train_df.columns)
        if (len(X) == win_lenght):
            hdddm.add_new_batch(X_df)
            result = hdddm.drift_detected
            if result:
                results['detected_drift_points'].append(i)
            else:
                pass
            i += win_lenght
    bar.finish()

    return results


def SEA_stream():
    train_X = pd.read_csv('../data/SEA/SEA_training_data.csv')
    train_y = pd.read_csv('../data/SEA/SEA_training_class.csv')
    train_stream = DataStream(train_X, train_y)

    test_X = pd.read_csv('../data/SEA/SEA_testing_data.csv')
    test_y = pd.read_csv('../data/SEA/SEA_testing_class.csv')
    test_stream = DataStream(test_X, test_y)

    return train_stream, test_stream


if __name__ == "__main__":
    random = 123
    n_times = 200
    w = 2000
    test_fraction = 0.25
    gammas = [0.5, 1.0, 1.5, 2.0]
    alfa = 0.1
    errors = []

    # No drift detection
    print('No drift detection')
    train_stream, test_stream = SEA_stream()

    model_error = []
    model = GaussianNB()

    for i in range(n_times):
        train_X, train_y = train_stream.next_sample(250)
        model.partial_fit(train_X, train_y, classes=[1, 2])

        test_X, test_y = test_stream.next_sample(250)
        model_preds = model.predict(test_X)
        model_error.append(np.mean(np.abs(test_y - model_preds)))

        i += 1
    errors.append(model_error)

    # HDDDM with different gammas
    for gamma in gammas:
        train_stream, test_stream = SEA_stream()
        print('Gamma {}'.format(gamma))
        model_error = []
        model = GaussianNB()

        train_X, train_y = train_stream.next_sample(250)
        model.partial_fit(train_X, train_y, classes=[1, 2])
        test_X, test_y = test_stream.next_sample(250)
        model_preds = model.predict(test_X)
        model_error.append(np.mean(np.abs(test_y - model_preds)))
        train_X_df = pd.DataFrame(train_X)
        hdddm = HDDDM(train_X_df, gamma)

        for i in range(n_times - 1):

            train_X, train_y = train_stream.next_sample(250)
            train_X_df = pd.DataFrame(train_X)
            hdddm.add_new_batch(train_X_df)
            result = hdddm.drift_detected

            if result:  # Reset ML model
                print("DRIFT")
                model = GaussianNB()
                model.partial_fit(train_X, train_y, classes=[1, 2])

                test_X, test_y = test_stream.next_sample(250)

                model_preds = model.predict(test_X)
                model_error.append(np.mean(np.abs(test_y - model_preds)))
            else:  # No reset
                model.partial_fit(train_X, train_y)

                test_X, test_y = test_stream.next_sample(250)

                model_preds = model.predict(test_X)
                model_error.append(np.mean(np.abs(test_y - model_preds)))

            i += 1
        errors.append(model_error)

    # HDDDM with alpha
    print('Alpha 0.5')

    model_error = []
    model = GaussianNB()
    train_stream, test_stream = SEA_stream()
    train_X, train_y = train_stream.next_sample(250)
    model.partial_fit(train_X, train_y, classes=[1, 2])
    test_X, test_y = test_stream.next_sample(250)
    model_preds = model.predict(test_X)
    model_error.append(np.mean(np.abs(test_y - model_preds)))

    train_X_df = pd.DataFrame(train_X)
    hdddm = HDDDM(train_X_df, gamma=None, alpha=0.5)

    for i in range(n_times - 1):

        train_X, train_y = train_stream.next_sample(250)
        train_X_df = pd.DataFrame(train_X)
        hdddm.add_new_batch(train_X_df)
        result = hdddm.drift_detected

        if result:  # Reset ML model
            print("DRIFT")
            model = GaussianNB()
            model.partial_fit(train_X, train_y, classes=[1, 2])

            test_X, test_y = test_stream.next_sample(250)
            model_preds = model.predict(test_X)
            model_error.append(np.mean(np.abs(test_y - model_preds)))
        else:  # No reset
            model.partial_fit(train_X, train_y)

            test_X, test_y = test_stream.next_sample(250)
            model_preds = model.predict(test_X)
            model_error.append(np.mean(np.abs(test_y - model_preds)))

        i += 1
    errors.append(model_error)

    # Plot

    errors = np.array(errors)
    errors = errors.T
    cols = ["No update", "Gamma 0.5", "Gamma 1.0", "Gamma 1.5", "Gamma 2.0", "Alpha 0.5"]

    errors_df = pd.DataFrame(errors, columns=cols)

    plt.figure(figsize=[16, 12])
    sns.lineplot(data=errors_df)

    plt.show()
