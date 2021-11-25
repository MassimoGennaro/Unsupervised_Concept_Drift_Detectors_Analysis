import numpy as np
from .HSIC import hsic_gam
from .lsit import test_independence
from progress.bar import IncrementalBar


class DAWIDD:

    def __init__(self, max_window_size=2500, min_window_size=100, min_p_value=0.1):
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.min_p_value = min_p_value

        self.X = []
        self.n_items = 0
        self.min_n_items = self.min_window_size / 4.

        self.drift_detected = False

    # You have to overwrite this function if you want to use a different test for independence
    def _test_for_independence(self):

        t = np.array(range(self.n_items)) / (1. * self.n_items)
        t /= np.std(t)
        t = t.reshape(-1, 1)

        return 1.0 if test_independence(np.array(self.X), t) == True else 0.0

    def add_record(self, x):
        self.drift_detected = False

        # Add item
        self.X.append(x.flatten())
        self.n_items += 1

        if self.n_items == self.max_window_size:
            # Test for drift
            print(" - Testing for drift")
            p = self._test_for_independence()

            if p <= self.min_p_value:
                self.drift_detected = True

            # Remove samples
            while self.n_items > self.min_window_size:
                # Remove old samples after min window size (baseline data never removed)
                self.X.pop(self.min_window_size)
                self.n_items -= 1

    def detected_change(self):
        return self.drift_detected


def dawidd_inference(train_results, max_window_size=2500, min_window_size=100, min_p_value=0.05):
    n_train = train_results["n_train"]
    stream = train_results["Stream"]
    drift_point = train_results["drift_point"]

    stream.restart()
    stream.next_sample(n_train)

    dawidd = DAWIDD(max_window_size, min_window_size, min_p_value)

    results = {'detected_drift_points': []}

    i = n_train
    bar = IncrementalBar('DAWIDD_inference', max=stream.n_remaining_samples())

    while stream.has_more_samples():
        bar.next()

        X = stream.next_sample()[0]
        dawidd.add_record(X)
        if dawidd.detected_change():
            print('Drift detected!')
            results['detected_drift_points'].append(i)
            if i > drift_point:
                return results
        else:
            pass
        i += 1
    bar.finish()

    return results
