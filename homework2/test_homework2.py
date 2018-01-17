import unittest
import numpy
from numpy import array, ones, diag, arange
from numpy.linalg import norm
from numpy.random import randn
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.linalg import solve as scipy_solve

# import the Python wrappers defined in homework2.wrappers
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

# these might be useful for your tests
import numpy
from numpy import array, ones, diag, arange
from numpy.linalg import norm
from numpy.random import randn
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.linalg import solve as scipy_solve

# this also might be useful for your tests
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
        x = array([1,2,3], dtype=numpy.double)
        y = array([4,5,6], dtype=numpy.double)
        z = vec_add(x,y)
        error = norm(z - (x+y))
        self.assertAlmostEqual(error, 0)


    def test_vec_sub(self):
        x = array([1,2,3], dtype=numpy.double)
        y = array([4,5,6], dtype=numpy.double)
        z = vec_sub(x,y)
        error = norm(z - (x-y))
        self.assertAlmostEqual(error, 0)
        
    def test_mat_add(self):
        x = array([[1,4],[2,4]], dtype=numpy.double)
        y = array([[2,6],[4,9]], dtype=numpy.double)
        z = mat_add(x,y)
        print 'mat add',z
        error = norm(z-(x+y))
        self.assertAlmostEqual(error,0)


    def test_mat_vec(self):
        x = array([[1,2,4],[2,6,4],[2,3,4]], dtype=numpy.double)
        y = array([3,4,2], dtype=numpy.double)
        z = mat_vec(x,y)
        print 'mat-vec',z
        error = norm(z-(numpy.dot(x,y)))
        self.assertAlmostEqual(error,0)
        
    def test_mat_mat(self):
        A = randn(8,8)  # create two random matrices
        B = randn(8,8)
        C = mat_mat(A,B)  # product using C code (wrapped by homework2.mat_mat)
        C_actual = numpy.dot(A,B)  # product using Numpy
        error = norm(C - C_actual)
        self.assertLess(error, 1e-8)  # account for floating point error


class TestSolver(unittest.TestCase):
    def test_solve_upper_triangular(self):
        u=array([[3,2,1],[0,5,1],[0,0,2]], dtype=numpy.double)
        b=array([13,7,4],dtype=numpy.double)
        x=solve_upper_triangular(u,b)
        print'x=',x
        x_actual=scipy_solve_triangular(u,b)
        print'actual',x_actual

    def test_solve_uptri_di(self):
        u=array([[1,0,0],[0,1,0],[0,0,1]], dtype=numpy.double)
        b=array([3,4,5], dtype=numpy.double)
        x=solve_upper_triangular(u,b)
        print'x=',x
        x_actual=scipy_solve_triangular(u,b)
        print'actual',x_actual

    def test_solve_lower_triangular(self):
        l=array([[3,0,0],[6,4,0],[8,2,4]],dtype=numpy.double)
        b=array([3,26,42],dtype=numpy.double)
        x=solve_lower_triangular(l,b)
        print"x=",x
        x_actual=scipy_solve(l,b)
        print'x_actual=',x_actual
        
class TestJacobi(unittest.TestCase):
    def test_jacobi(self):
        A = array([[2.0,1.0],[5.0,7.0]])
        b = array([11.0,13.0])
        v,x = jacobi(A,b)
        print v,x
#class TestGauss(unittest.TestCase):        
    def test_gauss(self):
        A = array([[2.0,1.0],[5.0,7.0]])
        b = array([11.0,13.0])
        v,x = gauss_seidel(A,b)
        print v,x   
class test_di(unittest.TestCase):
    def test_gauss_seidel_diag_5(self):       
        A = array([[ 5.0,-1.0,-1.0, 0.0, 0.0],[-1.0, 5.0,-1.0,-1.0, 0.0],[-1.0,-1.0, 5.0,-1.0,-1.0],[ 0.0,-1.0,-1.0, 5.0,-1.0],[ 0.0, 0.0,-1.0,-1.0, 5.0]])        
        b = array([0.0,1.0,2.0,3.0,4.0])        
        v, x = jacobi(A, b)
        print v,x
        
def time_gauss_seidel(n, number=1):
    """Returns the amount of time (in seconds) for gauss_seidel to execute for the
    given system size `n`.

    Feel free to use to timing purposes.

    Parameters
    ----------
    n : int
        The size of the test system to solve.
    number : int
        (Optional) The number of times to run the test. Useful for computing an
        average runtime.

    Returns
    -------
    avg_time : float
        The time, in seconds, it took to run gauss_seidel().
    """
    from timeit import timeit

    # write the setup (non-timed but necessary) code as a string
    s = '''
from numpy.random import randn
from homework2 import gauss_seidel
from test_homework2 import five_diagonal_system
N = %d
A,b = five_diagonal_system(N)
'''%(n)
    total_time = timeit('gauss_seidel(A,b)', setup=s, number=number)
    avg_time = total_time / number
    return avg_time


if __name__ == '__main__':
    #print '\n===== Timings ====='
    #n = 2**8
    #t = time_gauss_seidel(n)
    #print 'gauss_seidel(%d): %f'%(n, t)

    print '\n===== Running Tests ====='
    unittest.main(verbosity=2)
