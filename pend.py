import sys
import numpy as np
from scipy.integrate import odeint

# The gravitational acceleration (m.s-2).
g = 9.81


def deriv(y, t, L1, L2, m1, m2):
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) - (m1 + m2) * g * np.sin(
        theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (
                L1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) + m2 * L2 * z2 ** 2 * s * c) / L2 / (
                        m1 + m2 * s ** 2)
    return theta1dot, z1dot, theta2dot, z2dot


def solve(L1, L2, m1, m2, tmax, dt, y0):
    t = np.arange(0, tmax + dt, dt)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

    # Unpack z and theta as a function of time
    theta1, theta2 = y[:, 0], y[:, 2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # print(theta1[-1], theta2[-1])
    return theta1[-1], theta2[-1], x1, y1, x2, y2


def gen_theta1_init_theta1_init(theta_resolution):
    search_space = np.linspace(0, 2 * np.pi, theta_resolution)
    for theta1_init in search_space:
        for theta2_init in search_space:
            yield theta1_init, theta2_init


def simulate_pendulum(theta_resolution):
    # Pendulum rod lengths (m), bob masses (kg).
    L1, L2 = 1, 1
    m1, m2 = 1, 1

    # Maximum time, time point spacings and the time grid (all in s).
    tmax, dt = 30, 0.001

    for theta1_init, theta2_init in gen_theta1_init_theta1_init(theta_resolution):
        y0 = np.array([theta1_init, 0, theta1_init, 0])
        theta1, theta2, x1, y1, x2, y2 = solve(L1, L2, m1, m2, tmax, dt, y0)
        yield theta1_init, theta2_init, theta1, theta2


def main():
    for theta1_init, theta2_init, theta1, theta2 in simulate_pendulum(15):
        print("%f, %f, %f, %f" % (theta1_init, theta2_init, theta1, theta2))


if __name__ == "__main__":
    main()