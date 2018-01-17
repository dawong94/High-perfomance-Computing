#ifndef __homework2_solver_h
#define __homework2_solver_h

#include "linalg.h"
/*solve_lower_triangular
 Given matrix L, vector b, and integer N, L is an array of size NxN
 b is an array of length N. N is an integer. This implementation is to solve Lx=b where L
 is an lower triangular matrix and store all the values in out where it is 
 NxN size of matrix
 
 This function is return by reference.
 Parameters
  ----------
  out : double*
        storage for output matrix
  L : double* 
      The input matrices.
  b : double*
      The input vectors.
  
  N : int
  
      Dimensions of the input matrix and vector. 
      `L` is `N`x`N` and `b` is `N`x`1`.

  Returns
  -------
  out : double*
      The output matrix. Return by reference.
 */
void solve_lower_triangular(double* out, double* L, double* b, int N);

/*solve_upper_triangular
 Given matrix U, vector b, and integer N, U is an array of size NxN
 b is an array of length N. N is an integer. This implementation is to solve Ux=b where u
 is an upper triangular matrix and store all the values in out where it is 
 NxN size of matrix
 
 This function is return by reference.
 Parameters
  ----------
  out : double*
        storage for output matrix
  U : double* 
      The input matrices.
  b : double*
      The input vectors.
  
  N : int
  
      Dimensions of the input matrix and vector. 
      `U` is `N`x`N` and `b` is `N`x`1`.

  Returns
  -------
  out : double*
      The output vector. Return by reference.
 */
void solve_upper_triangular(double* out, double* U, double* b, int N);

/*decompose 
given matrices D,L,U and A of dimensions 'N'x'N' and N is the size of D,L,U and A
.D,L,U and A are all arrays of size 'N'x'N'. This implementation is to decompose matrix 
A into matrices D,L and U where D is an diagonal matrix, L is an lower triangular mtrix and
U is an upper triangular matrix without diagonal elements.
 
  Parameters
  ----------
  D : double*
       storage for output matrix
  L : double* 
      storage for output matrix
  U : double*
      storage for output matrix
  A : double*
      The input matrix
  N : int
      Dimensions of the input and output matrices  
      `U`,'D','L' and 'A' is `N`x`N` 

  Returns
  -------
  out : double*
      The output vector. Return by reference.

*/
void decompose(double* D, double* L, double* U, double* A, int N);

/*copy 
 this is a helper method which copys all the elements 
 from the xkp1(out) into xk. Given vectors xk and out which are
 arrays length of N.
 
 Parameters
  ----------
  xk : double*
       storage for output vector
  out : double*
       input vector
  N : int
      Dimensions of the input and output vectors  
      `xk` and 'out' is `N`x`1` 

  Returns
  -------
  xk : double*
      The output vector. Return by reference.

*/
void copy(double* xk,double* out,int N);

/*jacobi_step
    Given a guess xk for all real numbers
    and compute the next iterate xk1 by using
    jaboci method.D is diagnoal matrix, T is the 
    sum of L and U matrix, K is vector where is the 
    product of matrix T and vector xk. P is a vector where
    the vector b minus vector K.
    Parameters:
    __________
    out:double*
       storage for output vector 
    D:double*
    b:double*
    xk:double*
    P:double*
    T:double*
    K:double*
      input matrices and vectors
    N:int
        Dimensions of the input matrices and vectors

    Returns
    ________
    xk:double*
    next iterate xkp1 (out)
*/
void jacobi_step(double* out, double* D,double* T,double* K, double* P, int N, double* xk, double* b);

/* jacobi 
jacobi returns the number of iterations by value as an `int`. (it also
 returns the solution vector by reference as `out`).  Given matrix A with size NxN
 , a vector b with length N and size of matrix N and a epsilon.
 Parameters:
    __________
    out:double*
       storage for output vector 
    A:double*
    b:double*
      input matrices and vectors
    N:int
        Dimensions of the input matrices and vectors
    epsilon: double
        tolerance
    Returns
    ________
    out:double*
       output matrix (out)
    itr:int
       iteration times
*/ 
int jacobi(double* out, double* A, double* b, int N, double epsilon);

/*Gauss_step
    Given a guess xk for all real numbers
    and compute the next iterate xk1 by using
    Gauss_seidel method.S is sum of upper and diagnoal matrix,
    K is vector where is the 
    product of matrix S and vector xk. P is a vector where
    the vector b minus vector K.
    Parameters:
    __________
    out:double*
       storage for output vector 
    S:double*
    b:double*
    xk:double*
    P:double*
    L:double*
    K:double*
      input matrices and vectors
    N:int
        Dimensions of the input matrices and vectors

    Returns
    ________
    xk:double*
    next iterate xkp1 (out)
*/
void gauss_step(double* out, double* S,double* K, double* P, int N, double* xk, double* b, double* L);

/* gauss_seidel returns the number of iterations by value as an `int`. (it also
 returns the solution vector by reference as `out`). Given matrix A with size N*N
 b array vector with length N
  Parameters:
    __________
    out:double*
       storage for output vector 
    A:double*
    b:double*
      input matrices and vectors
    N:int
        Dimensions of the input matrices and vectors
    epsilon: double
        tolerance
    Returns
    ________
    out:double*
       output matrix (out)
    itr: int
    iteration times*/
int gauss_seidel(double* out, double* A, double* b, int N, double epsilon);

#endif
