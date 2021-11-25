import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor,\
    ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Lasso
from skmultiflow.drift_detection.adwin import ADWIN


from progress.bar import IncrementalBar


class Model:

    def __init__(self, sel_model):
        models = {

            'ExtraTreeClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'NaiveBayes': GaussianNB(),
            'LogisticRegression': LogisticRegression(),
            'ExtraTreeRegressor': ExtraTreesRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'Lasso': Lasso()

        }
        self.ml_model = models[sel_model]
        self.regression_models = ['ExtraTreeRegressor', 'RandomForestRegressor', 'Lasso']
        if sel_model in self.regression_models:
            self.regression = True
        else:
            self.regression = False

    def fit(self, X, y):
        self.ml_model.fit(X, y)

    def predict(self, X):
        preds = self.ml_model.predict(X)
        return preds

    def predict_proba(self, X):
        probs = self.ml_model.predict_proba(X)
        return probs


class Teacher(Model):
    pass


class Student(Model):
    pass


def teacher_student_train(teacher, student, stream, fit=True, train_perc=0.6):

    n_train = int(train_perc * stream.data.shape[0])
    X_train, y_train = stream.next_sample(n_train)

    if fit:
        teacher.fit(X_train, y_train)
        y_hat_train = teacher.predict(X_train)

        student.fit(X_train, y_hat_train)

    train_results = {"Teacher": teacher, "Student": student, "n_train": n_train, "Stream": stream, "X_train": X_train}

    return train_results


def teacher_student_inference(train_results):

    teacher = train_results["Teacher"]
    student = train_results["Student"]
    n_train = train_results["n_train"]
    stream = train_results["Stream"]
    X_train = train_results["X_train"]


    adwin = ADWIN()
    results = {'detected_drift_points': []}

    stream.restart()
    stream.next_sample(n_train)

    bar = IncrementalBar('ST_inference', max=stream.n_remaining_samples())
    i = n_train
    while stream.has_more_samples():
        bar.next()

        Xi, yi = stream.next_sample()
        """
        y_hat_teacher = teacher.predict(Xi)
        y_hat_student = student.predict(Xi)

        student_error = int(y_hat_teacher != y_hat_student)
        """

        if teacher.regression is True:
            y_hat_teacher = teacher.predict(Xi)[0]
            y_hat_student = student.predict(Xi)[0]

        else:
            probs_teacher = teacher.predict_proba(Xi)[0]

            if len(probs_teacher) < 2:
                y_hat_teacher = probs_teacher[0]
                y_hat_student = student.predict_proba(Xi)[0][0]

            elif len(probs_teacher) == 2:

                y_hat_teacher = np.max(probs_teacher)
                class_idx = np.argmax(probs_teacher)
                y_hat_student = student.predict_proba(Xi)[0][class_idx]


            else:  # more than two classes
                y_hat_teacher = np.max(probs_teacher)
                class_idx = np.argmax(probs_teacher)
                y_hat_student = student.predict_proba(Xi)[0][class_idx]
        
        student_error = np.abs(y_hat_teacher - y_hat_student)
        adwin.add_element(student_error)

        if adwin.detected_change():
            results['detected_drift_points'].append(i)


        else:
            pass

        i += 1
    bar.finish()

    return results
