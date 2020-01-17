from datetime import datetime
import os
import csv
from celery import chord
from celery.exceptions import Reject
import numpy as np
from ..app import app
from .worker import solve


@app.task
def writeToCsv(results):
    file = app.conf.RESULTS_DIR + "/distributed.csv"
    with open(file, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|')
        writer = csv.DictWriter(file, fieldnames=["theta1_init", "theta2_init", "theta1", "theta2"])
        writer.writeheader()
        for r in results:
            t1, t2, theta1, theta2, x1, y1, x2, y2 = r
            writer.writerow({'theta1_init': t1, 'theta2_init': t2, 'theta1': theta1[len(theta1) - 1],
                             'theta2': theta2[len(theta2) - 1]})


def get_experiment_status_filename(status):
    return os.path.join(app.conf.STATUS_DIR, status)


def get_experiment_status_time():
    """Get the current local date and time, in ISO 8601 format (microseconds and TZ removed)"""
    return datetime.now().replace(microsecond=0).isoformat()


@app.task
def record_experiment_status(status):
    with open(get_experiment_status_filename(status), 'w') as fp:
        fp.write(get_experiment_status_time() + '\n')


## Seeding the computations

def gen_simulation_model_params(theta_resolution):
    search_space = np.linspace(0, 2 * np.pi, theta_resolution)
    for theta1_init in search_space:
        for theta2_init in search_space:
            yield theta1_init, theta2_init, app.conf.PDAJ_L1, \
                  app.conf.PDAJ_L2, app.conf.PDAJ_M1, app.conf.PDAJ_M2, app.conf.PDAJ_TMAX, app.conf.PDAJ_DT, \
                  np.array([theta1_init, 0, theta2_init, 0])


@app.task
def seed_computations(ignore_result=True):
    if os.path.exists(get_experiment_status_filename('started')):
        raise Reject('Computations have already been seeded!')
    record_experiment_status.si('started').delay()
    chord(
        (solve.s(theta1_init, theta2_init, L1, L2, M1, M2, TMAX, DT, y0)
         for theta1_init, theta2_init, L1, L2, M1, M2, TMAX, DT, y0 in
         gen_simulation_model_params(app.conf.PDAJ_RESOLUTION)),
        (writeToCsv.s())
    ).delay()
    record_experiment_status.si('completed').delay()
