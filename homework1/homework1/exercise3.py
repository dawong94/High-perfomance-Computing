# Hint: you should only need the following functions from numpy and scipy
from numpy import diag, tril, triu, dot, array
from numpy.linalg import norm
from scipy.linalg import solve_triangular

def decompose(A):
    """Helper function which retruns the
    D,L and U matrices given a matrix A.
    D,L and U are diagnal matrix, lower triangular
    and upper triangualr matrix without diagonal elements
    Parameter:
    ___________
    A: numpy.ndarray


    Returns:
    ____________
    D:numpy.ndarray
    L:numpy.ndarray
    U:numpy.ndarray

    D,L and U are diagnal matrix, lower triangular
    and upper triangualr matrix without diagonal elements
    """
    D_temp = diag(A)
    D= diag(D_temp)
    L = tril(A)- D
    U = triu(A) - D
    return D,L,U

def is_sdd(A):
    """ helper function which returns True
    if A is strictly diagonally dominant and False otherwise

    Parameters:
    ____________
    A: numpy.ndarray

    Return:
    ______
    bool
    true or false
    """
    suma= 0
    for i in range(len(A)):
        for k in range(len(A)):
            if i != k :
                suma = suma + abs(A[i,k])

        if abs(A[i,i]) < suma :
            return False
        suma = 0

    return True

def jacobi_step(D, L, U, b, xk):
    """Given a guess xk for all real numbers
    and compute the next iterate xk+1 by using
    jaboci method.
    Parameters:
    __________
    D:numpy.ndarray
    L:numpy.ndarray
    U:numpy.ndarray
    b:numpy.ndarray
    xk:numpy.ndarray

    Return:
    ________
    xk1:numpy.ndarray
    next iterate xk+1
    """
    T= L+U
    S= diag(D)
    xk1 = (b -dot(T,xk))/S
    return xk1

def jacobi_iteration(A, b, x0, epsilon=1e-8):
    """given a guess x0 returns an approximate solution to the system Ax = b.
    calling functions decompose and jacobi_step to help write this function.

    Parameter:
    ___________
    A:numpy.ndarray
    x0:numpy.ndarray
    b:numpy.ndarray
    epsilon:float

    Rerurn:
    _______
    xk1:numpy.ndarray
    return the iterate xk+1 which is the final solution
    """

    D,L,U = decompose(A)
    xk =x0
    xk1 = jacobi_step(D, L, U, b, xk)

    while norm(xk1 - xk) > epsilon:
    #for i in range(25):
        xk = xk1
        xk1 = jacobi_step(D, L, U, b, xk)
        #print xk1

    return xk1

def gauss_seidel_step(D, L, U, b, xk):
    """Given a guess xk for all real numbers
    and compute the next iterate xk+1 by using
    Gauss_seidel method.
    Paramaters:
    ___________
    D:numpy.ndarray
    L:numpy.ndarray
    U:numpy.ndarray
    b:numpy.ndarray
    xk:numpy.ndarray

    Return:
    _______
    xk1:numpy.ndarray
    next iterate xk+1
    """

    S= D+U
    xk1 = solve_triangular(S,(b - dot(L,xk)))

    return xk1

def gauss_seidel_iteration(A, b, x0, epsilon=1e-8):
    """given a guess x0 returns an approximate solution to the system Ax = b.
    calling functions decompose and gauss_seidel_step to help write this function.

    Parameter:
    ___________
    A:numpy.ndarray
    x0:numpy.ndarray
    b:numpy.ndarray
    epsilon:float

    Rerurn:
    _______
    xk1:numpy.ndarray
    return the iterate xk+1 which is the final solution
    """
    D,L,U = decompose(A)
    xk =x0
    xk1 = gauss_seidel_step(D, L, U, b, xk)
    while norm(xk1 - xk) > epsilon:
        xk = xk1
        xk1 = gauss_seidel_step(D, L, U, b, xk)

    return xk1

