# Datasets Loading
from datasetloader.load_dataset import load_stream

# Drift Methods
from studentteacher.student_teacher import *
from d3.D3 import *
from hdddm.HDDDM import *
from dawidd.DAWIDD import *

# Performances Computation
from performance.bifet_metrics import compute_bifet_metrics

# Utilities
import json
from progress.bar import IncrementalBar
import time
import os


def run_experiments(datasets, models, n_repetitions=5):
    print('START')
    print('-----')
    start_time = time.time()
    for dataset_name in datasets:

        print("Starting with dataset: {}".format(dataset_name))
        streams = []
        real_drift_points = []

        # Drift Injection and Stream Creation
        for i in range(n_repetitions):
            dstream, drifted_rows, drift_point = load_stream(dataset_name, shuffle=False)
            streams.append(dstream)
            real_drift_points.append(drift_point)
        print(real_drift_points)

        # Training Phase
        if dataset_name in ['anas']:
            teacher = Teacher('RandomForestRegressor')
            student = Student('RandomForestRegressor')
        else:
            teacher = Teacher('RandomForestClassifier')
            student = Student('RandomForestClassifier')

        train_results = []
        bar = IncrementalBar('Training Phase', max=len(streams))
        for s in streams:
            bar.next()
            if 'student-teacher' in models:
                print(' - Fitting ST')
                train_results.append(teacher_student_train(teacher, student, s, fit=True))
            else:
                print(' - Not fitting ST')
                train_results.append(teacher_student_train(teacher, student, s, fit=False))
        bar.finish()

        # Detection Phase
        inf_results = {m: [] for m in models}

        inference_functions = {'student-teacher': teacher_student_inference,
                               'd3': d3_inference,
                               'hdddm': hdddm_inference,
                               'dawidd': dawidd_inference}

        i = 1
        for idx, r in enumerate(train_results):
            r['drift_point'] = real_drift_points[idx]
            print("Iteration {}".format(i))
            for m in models:
                inf_results[m].append(inference_functions[m](r))
            i += 1

        # Performances Computation
        detected_drifts = {m: [] for m in models}

        for m in inf_results.keys():
            for ele in inf_results[m]:
                detected_drifts[m].append(ele['detected_drift_points'])

        performances = compute_bifet_metrics(detected_drifts, real_drift_points, models)

        # Performances Saving
        if not os.path.exists('results'):
            os.mkdir('results')

        with open('results/' + dataset_name + '_bifet_performances.json', 'w') as f:
            json.dump(performances, f)

        print('Performance saved on results/' + dataset_name + '_bifet_performances.json')
        print('Finished dataset {}'.format(dataset_name))

    print(f"Total time: {(time.time() - start_time) / 60} minutes")
    print('---')
    print('END')


if __name__ == '__main__':
    # Setup
    datasets = ['electricity', 'weather', 'forestcover', 'anas']
    models = ['hdddm', 'd3', 'dawidd', 'student-teacher']
    n_repetitions = 50

    run_experiments(datasets, models, n_repetitions)
