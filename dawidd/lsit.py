# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from .HSIC import hsic_gam
# from mutual_info import mutual_information


class LeastSquaresIndependenceTest(RegressorMixin):
    def __init__(self, y_type, C, sigma, X_prototypes, Y_prototypes):
        self.y_type = y_type
        self.C = C
        self.sigma = sigma  # sigma squared
        self.X_prototypes = X_prototypes
        self.Y_prototypes = Y_prototypes

        self.alpha = None

        self.H = None  # Cache
        self.Phis = None
        self.Phi = None
        self.Phi_test = None

    def get_params(self, deep=True):
        return {"y_type": self.y_type, "C": self.C, "sigma": self.sigma, "X_prototypes": self.X_prototypes,
                "Y_prototypes": self.Y_prototypes}

    def set_params(self, **params):
        if "y_type" in params.keys():
            self.y_type = params["y_type"]
        if "C" in params.keys():
            self.C = params["C"]
        if "sigma" in params.keys():
            self.sigma = params["sigma"]
        if "X_prototypes" in params.keys():
            self.X_prototypes = params["X_prototypes"]
        if "Y_prototypes" in params.keys():
            self.Y_prototypes = params["Y_prototypes"]

    def __basis_function(self, X, Y):
        phi_x = np.hstack(
            [np.exp((-.5 / self.sigma) * np.sum(np.square(X - p), axis=1)).reshape(-1, 1) for p in self.X_prototypes])

        phi_y = None
        if self.y_type == 0:
            phi_y = np.hstack([np.exp((-.5 / self.sigma) * np.sum(np.square(Y - p), axis=1)).reshape(-1, 1) for p in
                               self.Y_prototypes])
        else:
            phi_y = np.apply_along_axis(lambda x: [x == p for p in self.Y_prototypes], axis=1, arr=Y).reshape(
                Y.shape[0], len(self.Y_prototypes))  # TODO: More efficient implementation

        return np.multiply(phi_x, phi_y)

    def predict(self, X_train, Y_train):
        n = X_train.shape[0]

        s = 0.0
        for phi in self.Phis:
            s += np.sum(np.square(np.dot(phi, self.alpha)))

        s *= -1. / (2. * n ** 2)

        s += np.mean(np.dot(self.Phi, self.alpha), axis=0)

        s -= 0.5

        return float(s)

    def fit(self, X_train, Y_train):
        n = X_train.shape[0]
        dX = X_train.shape[1]
        dY = Y_train.shape[1]
        n_proto = len(self.X_prototypes)

        H = np.zeros((n_proto, n_proto))
        if self.Phis is None:
            self.Phis = []
            for i in range(n):
                Y_ = np.repeat(Y_train[i, :], repeats=n).reshape(n, dY)
                phi = self.__basis_function(X_train, Y_)
                self.Phis.append(phi)

                H += np.sum(np.apply_along_axis(lambda x: np.outer(x, x), axis=1, arr=phi),
                            axis=0)  # TODO: More efficient implementation!
            H *= (1. / (n ** 2))
            self.H = H
        else:
            H = self.H

        if self.Phi is None:
            self.Phi = self.__basis_function(X_train, Y_train)
        h = np.mean(self.Phi, axis=0)

        self.alpha = np.dot(np.linalg.inv(H + self.C * np.eye(h.shape[0])), h)

    def score(self, X_test, Y_test):
        n = X_test.shape[0]

        Phi_test = self.Phi_test
        if self.Phi_test is None:
            Phi_test = self.__basis_function(X_test, Y_test)
            self.Phi_test = Phi_test

        s = -1. * np.mean(np.dot(Phi_test, self.alpha), axis=0)
        s += (1. / (2. * n)) * np.sum(np.square(np.dot(Phi_test, self.alpha)), axis=0)

        return -1. * s  # Note: We want to minimize the score. However, GridSearchCV wants to maximize it!


def LSMI(X, Y, y_type, sigma_list, lambda_list, b=200, n_folds=3, verbose=False):
    # return mutual_information((X, Y), k=5)

    # """
    n = X.shape[0]
    b = min(n, b)

    # Gaussian centers are randomly chosen from samples
    X_prototypes = X
    Y_prototypes = Y
    if b < n:
        rand_index = np.random.permutation(n)
        X_prototypes = X[rand_index[:b], :]
        Y_prototypes = Y[rand_index[:b], :]

    # Optimize hyperparameters by grid search cross validation
    scores = np.zeros((len(sigma_list), len(lambda_list)))

    kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        # Split
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Grid search
        i = 0
        for sigma in sigma_list:
            H = None
            Phis = None
            Phi = None
            Phi_test = None

            j = 0
            for C in lambda_list:
                model = LeastSquaresIndependenceTest(y_type, C, sigma, X_prototypes, Y_prototypes)
                model.H = H  # Use caches matrices
                model.Phis = Phis
                model.Phi = Phi
                model.Phi_test = Phi_test
                model.fit(X_train, Y_train)

                scores[i][j] += model.score(X_test, Y_test)

                Phis = model.Phis  # Cache matrices
                H = model.H
                Phi = model.Phi
                Phi_test = model.Phi_test

                j += 1

            i += 1

    scores *= 1. / (n_folds)

    # Select best parameters
    sigma_idx, lambda_idx = np.unravel_index(scores.argmax(), scores.shape)
    best_sigma, best_C = sigma_list[sigma_idx], lambda_list[lambda_idx]
    if verbose is True:
        print(best_C, best_sigma)

    # Compute LSMI
    model = LeastSquaresIndependenceTest(y_type, best_C, best_sigma, X_prototypes, Y_prototypes)
    model.fit(X, Y)

    return model.predict(X, Y)
    # """


def LSIT(X, Y, y_type, T=10, b=10, fold=3, verbose=True, n_jobs=-1):
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples")

    n = X.shape[0]
    width_list = np.array([0.6, 0.8, 1., 1.2, 1.4])
    lambda_list = np.array([10. ** (-3.), 10. ** (-2.), 10. ** (-1.), 10. ** (0.), 10. ** (1.)])

    SMI0 = LSMI(X, Y, y_type, width_list, lambda_list, b, fold)

    SMI = Parallel(n_jobs=n_jobs)(
        delayed(LSMI)(X, np.random.permutation(Y), y_type, width_list, lambda_list, b, fold) for _ in range(T))
    SMI = np.array(SMI)

    pvalue = np.mean(SMI > SMI0)

    return {"pvalue": pvalue, "SMI0": SMI0, "SMI": SMI}  # p=0: Dependent, p=1: Independent


def test_independence(X, Y):
    testStat, thresh = hsic_gam(X, Y, alph=0.05)
    # print(testStat)
    # print(thresh)
    return testStat < thresh


if __name__ == "__main__":
    np.random.seed(2)

    # Create two regression data sets and join them (concept drift!)
    n = 50

    X1 = (np.random.rand(n, 1) * 2. - 1.) * 20.
    Y1 = -0.5 * X1 + 0.5
    X2 = (np.random.rand(n, 1) * 2. - 1.) * 20.
    Y2 = 0.5 * X2 + 0.5  # np.random.randn(n, 1) + np.sin(X2 / 20.*np.pi)

    data_stream_X = np.concatenate((X1, X2), axis=0)
    data_stream_Y = np.concatenate((Y1, Y2), axis=0)

    data_stream_X /= np.std(data_stream_X)
    data_stream_Y /= np.std(data_stream_Y)

    data_stream = np.hstack((data_stream_X, data_stream_Y))

    t = range(len(data_stream_X))
    # t -= np.mean(t)
    t /= np.std(t)
    t = t.reshape(-1, 1)
    # print(t)

    # Plot
    # import matplotlib.pyplot as plt
    # plt.scatter(data_stream_X, data_stream_Y)
    # plt.show()

    # Test for independence
    print(LSIT(data_stream_X, t, y_type=0, T=10))
    # print(test_independence(data_stream, t))
    # print(LSIT(data_stream_X, data_stream_Y, y_type=0, T=10))
    # print(test_independence(data_stream_X, data_stream_Y))
