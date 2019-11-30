import argparse
from multiprocessing import Pool
import numpy as np
from scipy.integrate import odeint
import csv

DEFAULT_RESOLUTION = 6
DEFAULT_TMAX = 30
DEFAULT_DT = 0.01
DEFAULT_L1 = 1
DEFAULT_L2 = 1
DEFAULT_M1 = 1
DEFAULT_M2 = 1

# The gravitational acceleration (m.s-2).
g = 9.81


def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
             (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
             m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)
    return theta1dot, z1dot, theta2dot, z2dot


def solve(L1, L2, m1, m2, tmax, dt, y0, t1init, t2init):
    t = np.arange(0, tmax + dt, dt)
    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    theta1, theta2 = y[:, 0], y[:, 2]
    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return theta1, theta2, x1, y1, x2, y2, t1init, t2init


def gen_simulation_model_params(theta_resolution):
    search_space = np.linspace(0, 2 * np.pi, theta_resolution)
    for theta1_init in search_space:
        for theta2_init in search_space:
            yield theta1_init, theta2_init, np.array([theta1_init, 0, theta2_init, 0])


def myFunc(args):
    return solve(DEFAULT_L1, DEFAULT_L2, DEFAULT_M1, DEFAULT_M2, DEFAULT_TMAX, DEFAULT_DT, args[2], args[0], args[1])


def simulate_pendulum(theta_resolution, dt=DEFAULT_DT, tmax=DEFAULT_TMAX, L1=DEFAULT_L1, L2=DEFAULT_L2, m1=DEFAULT_M1, m2=DEFAULT_M2):
    with Pool() as pool:
        # kao parametar prosledjena je funkcija koja predstavlja generator pocetnih uglova i y0
        results = pool.imap(myFunc, gen_simulation_model_params(theta_resolution), chunksize=theta_resolution)
        for res in results:
            # yildujem theta1,theta2,theta1_init i theta2_init
            yield res[0][-1], res[1][-1], res[6], res[7]


def calculateAndWrite():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'results_file',
        help="Filename where the results will be stored, in CSV format"
    )
    parser.add_argument(
        '-r',
        '--resolution',
        metavar='NUM',
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Resolution, %d by default" % DEFAULT_RESOLUTION
    )
    parser.add_argument(
        '--tmax',
        metavar='NUM',
        type=float,
        default=DEFAULT_TMAX,
        help="Simulation time, %f by default" % DEFAULT_TMAX
    )
    parser.add_argument(
        '--dt',
        metavar='NUM',
        type=float,
        default=DEFAULT_DT,
        help="Simulation time step, %f by default" % DEFAULT_DT
    )
    args = parser.parse_args()
    results = simulate_pendulum(
        theta_resolution=args.resolution,
        dt=args.dt,
        tmax=args.tmax,
    )
    writeToCsv(results, args.results_file)


def writeToCsv(results, file):
    with open(file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|')
        writer = csv.DictWriter(file, fieldnames=["theta1_init", "theta2_init", "theta1", "theta2"])
        writer.writeheader()
        for r in results:
            theta1, theta2, theta1_init, theta2_init = r
            writer.writerow({'theta1_init': theta1_init, 'theta2_init': theta2_init, 'theta1': theta1,
                             'theta2': theta2})
if __name__ == '__main__':
    calculateAndWrite()
