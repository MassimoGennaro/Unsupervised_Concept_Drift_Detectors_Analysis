import numpy as np
import math


def detect_time(detected_points, real_drift_point):

    dp = np.array(detected_points)
    dp_correct = dp[dp > real_drift_point]

    if len(dp_correct) > 0:
        first_correct_alarm = dp_correct[0]
        det_time = first_correct_alarm - real_drift_point
    else:
        det_time = np.nan

    return det_time


def mean_time_to_detect(detected_points, real_drift_points):
    det_times = []
    for i in range(len(detected_points)):
        det_time = detect_time(detected_points[i], real_drift_points[i])
        if not np.isnan(det_time):
            det_times.append(det_time)

    mean_mtd = np.mean(np.array(det_times))
    std_mtd = np.std(np.array(det_times))

    return mean_mtd, std_mtd


def mean_time_between_false_alarm(detected_point, real_drift_point):

    dp = np.array(detected_point)

    dp_fa = dp[dp < real_drift_point]

    if len(dp_fa) == 1:
        mtfa = dp_fa[0]
    elif len(dp_fa) == 0:  # No False Alarms
        mtfa = math.nan
    else:
        mtfa = np.nanmean(np.diff(dp_fa))

    return mtfa


def mean_mtfa(detected_points, real_drift_points):
    mtfas = []
    for i in range(len(detected_points)):

        mtfas.append(mean_time_between_false_alarm(detected_points[i], real_drift_points[i]))

    mean_mtfa = np.nanmean(np.array(mtfas))
    std_mtfa = np.nanstd(np.array(mtfas))

    return mean_mtfa, std_mtfa


def missed_detection(detected_point, real_drift_point):

    dp = np.array(detected_point)

    first_succ_alarm = dp[dp > real_drift_point]

    if len(first_succ_alarm) > 0:
        md_val = False
    else:
        md_val = True

    return md_val


def missed_detection_rate(detected_points, real_drift_points):

    md_count = 0
    for i in range(len(detected_points)):
        if missed_detection(detected_points[i], real_drift_points[i]):
            md_count += 1

    return md_count/(len(real_drift_points))


def compute_bifet_metrics(detected_points, real_drift_points, methods):

    results = {m: {'MTFA': [0, 0], 'MTD': [0, 0], 'MDR': 0} for m in methods}

    for m in methods:

        results[m]['MTFA'][0], results[m]['MTFA'][1] = mean_mtfa(detected_points[m], real_drift_points)
        results[m]['MTD'][0], results[m]['MTD'][1] = mean_time_to_detect(detected_points[m], real_drift_points)
        results[m]['MDR'] = missed_detection_rate(detected_points[m], real_drift_points)

    return results
