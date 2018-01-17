import numpy

from numpy import linspace, sin, exp, empty
from scipy.integrate import simps

from homework3 import (
    simps_parallel,
    time_simps_parallel,
)


if __name__ == '__main__':
    print '\n===== Running Timings ====='
    N = 2**28 + 1
    x = linspace(-1, 3, N)
    y = exp(sin(x))

    # Check the answer.
    ans1 = simps(y, x)
    ans2 = simps_parallel(y, x, num_threads=4)
    error = abs(ans1-ans2)
    print 'error from untimed run: ', error
    assert error < 1.e-10, 'error is too large'

    times = empty(5)
    for i in range(5):
        times[i] = time_simps_parallel(y, x, num_threads=4)

    print 'times: ', times
    print 'minimum time is: ', times.min()
