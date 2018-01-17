import unittest
import numpy
from numpy import array, ones, diag, arange
from numpy.linalg import norm
from numpy.random import randn
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.linalg import solve as scipy_solve

from homework2 import (
    vec_add,
    vec_sub,
    vec_norm,
    mat_add,
    mat_vec,
    mat_mat,
    solve_lower_triangular,
    solve_upper_triangular,
    jacobi,
    gauss_seidel,
)

def random_lower_triangular(n):
    L = randn(n,n)
    for i in range(n):
        for j in range(n):
            if i < j:
                L[i,j] = 0
    return L

def random_upper_triangular(n):
    U = randn(n,n)
    for i in range(n):
        for j in range(n):
            if i > j:
                U[i,j] = 0
    return U

def five_diagonal_system(n):
    """Returns the 5-diagonal matrix A and vector b used in some of the Jacobi and
    Gauss-Seidel tests.

    Parameters
    ----------
    n : int
        The size of the system.

    Returns
    -------
    A : array (matrix)
        The system matrix.
    b : array (vector)
        The system vector.
    """
    v = 5*ones(n)
    w1 = -ones(n-1)
    w2 = -ones(n-2)
    A = diag(v) + diag(w2,k=-2) + diag(w1,k=-1) + diag(w1,k=1) + diag(w2,k=2)
    b = arange(n)
    return A,b

class TestLinalg(unittest.TestCase):
    def test_vec_add(self):
        x = array([1,2,3])
        y = array([4,5,6])
        z = vec_add(x,y)
        error = norm(z - (x+y))
        self.assertAlmostEqual(error, 0)

    def test_vec_sub(self):
        x = array([1,2,3])
        y = array([4,5,6])
        z = vec_sub(x,y)
        error = norm(z - (x-y))
        self.assertAlmostEqual(error, 0)

    def test_vec_norm(self):
        x = randn(17)
        n_actual = norm(x)
        n = vec_norm(x)
        self.assertAlmostEqual(n_actual, n)

    def test_mat_add(self):
        A = randn(31,31)
        B = randn(31,31)
        C = mat_add(A,B)
        error = norm(C - (A+B))
        self.assertAlmostEqual(error, 0)

    def test_mat_vec(self):
        A = randn(13,13)
        x = randn(13)
        y_actual = A.dot(x)
        y = mat_vec(A,x)
        error = norm(y - y_actual)
        self.assertAlmostEqual(error, 0)

    def test_mat_mat(self):
        A = randn(29,29)
        B = randn(29,29)
        C = mat_mat(A,B)
        C_actual = numpy.dot(A,B)
        error = norm(C - C_actual)
        self.assertAlmostEqual(error, 0)

class TestSolver(unittest.TestCase):
    def test_solve_lower_triangular_diagonal(self):
        # create a random diagonal matrix
        L = diag(randn(23)) + 23*numpy.diag(numpy.ones(23))
        b = randn(23)
        y_actual = scipy_solve_triangular(L,b,lower=True)
        y = solve_lower_triangular(L,b)
        error = norm(y - y_actual)
        self.assertAlmostEqual(error, 0)

    def test_solve_lower_triangular(self):
        # create a random lower triangular matrix
        L = random_lower_triangular(23) + 23*numpy.diag(numpy.ones(23))
        b = randn(23)
        y_actual = scipy_solve_triangular(L,b,lower=True)
        y = solve_lower_triangular(L,b)
        error = norm(y - y_actual)
        self.assertAlmostEqual(error, 0)

    def test_solve_upper_triangular_diagonal(self):
        # create a random diagonal matrix
        U = diag(randn(23)) + 23*numpy.diag(numpy.ones(23))
        b = randn(23)
        y_actual = scipy_solve_triangular(U,b)
        y = solve_upper_triangular(U,b)
        error = norm(y - y_actual)
        self.assertAlmostEqual(error, 0)

    def test_solve_upper_triangular(self):
        # create a random upper triangular matrix
        U = random_upper_triangular(23) + 23*numpy.diag(numpy.ones(23))
        b = randn(23)
        y_actual = scipy_solve_triangular(U,b)
        y = solve_upper_triangular(U,b)
        error = norm(y - y_actual)
        self.assertAlmostEqual(error, 0)

    def test_jacobi(self):
        # create random sdd matrix
        A = randn(37,37) + 37*numpy.diag(numpy.ones(37))
        b = randn(37)
        y_actual = scipy_solve(A,b)
        y,_ = jacobi(A,b)
        error = norm(y - y_actual)
        self.assertLess(error, 1e-4)

    def test_jacobi_epsilon(self):
        # create random sdd matrix
        A = randn(37,37) + 37*numpy.diag(numpy.ones(37))
        b = randn(37)
        epsilon=1e-12
        y_actual = scipy_solve(A,b)
        y,_ = jacobi(A,b)
        error = norm(y - y_actual)
        self.assertLess(error, 1e-8)

    def test_gauss_seidel(self):
        # create random sdd matrix
        A = randn(37,37) + 37*numpy.diag(numpy.ones(37))
        b = randn(37)
        y_actual = scipy_solve(A,b)
        y,_ = gauss_seidel(A,b)
        error = norm(y - y_actual)
        self.assertLess(error, 1e-4)

    def test_gauss_seidel_epsilon(self):
        # create random sdd matrix
        A = randn(37,37) + 37*numpy.diag(numpy.ones(37))
        b = randn(37)
        epsilon=1e-12
        y_actual = scipy_solve(A,b)
        y,_ = gauss_seidel(A,b)
        error = norm(y - y_actual)
        self.assertLess(error, 1e-8)

    def test_jacobi_iteration_count(self):
        n = 10
        v = 5*ones(n)
        w1 = -ones(n-1)
        w2 = -ones(n-2)
        A = diag(v) + diag(w2,k=-2) + diag(w1,k=-1) + diag(w1,k=1) + diag(w2,k=2)
        b = arange(n)

        # we test the iteration count in a given range to account for some
        # numerical and (slight) algorithmic variability
        _, num_iterations = jacobi(A, b)
        self.assertTrue(55 < num_iterations < 75)

    def test_gauss_seidel_iteration_count(self):
        A,b = five_diagonal_system(10)

        # we test the iteration count in a given range to account for some
        # numerical and (slight) algorithmic variability
        _, num_iterations = gauss_seidel(A, b)
        self.assertTrue(25 < num_iterations < 45)

def time_lower_triangular(n, number=3):
    # returns the time to perform a random nxn lower triangular solve
    from timeit import timeit

    s = '''
from numpy.random import randn
from homework2 import solve_lower_triangular
from test_homework2 import random_lower_triangular
N = %d
L = random_lower_triangular(N)
b = randn(N)
'''%(n)
    total_time = timeit('solve_lower_triangular(L,b)', setup=s, number=number)
    avg_time = total_time / number
    return avg_time

def time_upper_triangular(n, number=3):
    # returns the time to perform a random nxn upper triangular solve
    from timeit import timeit

    s = '''
from numpy.random import randn
from homework2 import solve_upper_triangular
from test_homework2 import random_upper_triangular
N = %d
L = random_upper_triangular(N)
b = randn(N)
'''%(n)
    total_time = timeit('solve_upper_triangular(L,b)', setup=s, number=number)
    avg_time = total_time / number
    return avg_time

def time_jacobi(n, number=1):
    # returns the average time to perform a random nxn lower triangular solve
    from timeit import timeit

    s = '''
from numpy import diag, ones
from numpy.random import randn
from homework2 import jacobi
N = %d
A = randn(N,N) + N*diag(ones(N))
b = randn(N)
'''%(n)
    total_time = timeit('jacobi(A,b)', setup=s, number=number)
    avg_time = total_time / number
    return avg_time

def time_gauss_seidel(n, number=1):
    # returns the average time to perform a random nxn lower triangular solve
    from timeit import timeit

    s = '''
from numpy import diag, ones
from numpy.random import randn
from homework2 import gauss_seidel
N = %d
A = randn(N,N) + N*diag(ones(N))
b = randn(N)
'''%(n)
    total_time = timeit('gauss_seidel(A,b)', setup=s, number=number)
    avg_time = total_time / number
    return avg_time


def print_time_jacobi(n=2**10):
    t = time_jacobi(n)
    print '\n%f'%(t)

def print_time_gauss_seidel(n=2**10):
    t = time_gauss_seidel(n)
    print '\n%f'%(t)

if __name__ == '__main__':
    print '\n===== Timings ====='
    n = 2**10
    t = time_jacobi(n)
    print 'jacobi(%d): %f'%(n, t)

    t = time_gauss_seidel(n)
    print 'gauss_seidel(%d): %f'%(n, t)

    print '\n===== Running Tests ====='
    unittest.main(verbosity=2)
