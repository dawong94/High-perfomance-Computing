#ifndef __homework2_linalg_h
#define __homework2_linalg_h

/*
  vec_add

  Computes the sum of two vectors.

  Parameters
  ----------
  out : double*
    Storage for the resulting sum vector.
  v : double*
  w : double*
    The two vectors to sum.
  N : int
    The length of the vectors, `out`, `v`, and `w`.

  Returns
  -------
  out : double*
    (Output by reference.) The sum of `v` and `w`.
*/
void vec_add(double* out, double* v, double* w, int N);

/*
  vec_sub

  Computes the sub of two vectors.

  Parameters
  ----------
  out : double*
    Storage for the resulting sum vector.
  v : double*
  w : double*
    The two vectors to subtract.
  N : int
    The length of the vectors, `out`, `v`, and `w`.

  Returns
  -------
  out : double*
    (Output by reference.) The subtract of `v` and `w`.
*/
void vec_sub(double* out, double* v, double* w, int N);

/*
  vec_sub

  Computes the 2-norm of a vector.

  Parameters
  ----------
  
  v : double*
    The two vector to norm.
  N : int
    The length of the vector  `v`

  Returns
  -------
   type:int 
     
    The 2-NORM of `v` 
*/
double vec_norm(double* v, int N);

/*
  mat_add

  Computes the sub of two matrices.

  Parameters
  ----------
  out : double*
    Storage for the resulting sum matrices.
  A : double*
  B : double*
    The two matrices to add.
  N : int
  M : int
    The size of two matrices , `out`, `A`, and `B`.

  Returns
  -------
  out : double*
    (Output by reference.) The SUM of `A` and `B`.
*/
void mat_add(double* out, double* A, double* B, int M, int N);

/*
  mat_add

  Computes the product of matrix and vector

  Parameters
  ----------
  out : double*
    Storage for the resulting matrix multiplies vector.
  A : double*
    The  matrix to multiply.
  x ： double*
      THE vector to multiply
  N : int
  M : int
    The size of matrix , `out`, `A`
    and lenth of 'X' N

  Returns
  -------
  out : double*
    (Output by reference.) The dot product of `A` and `x`.
*/
void mat_vec(double* out, double* A, double* x, int M, int N);

// A is MxN, B is NxK, and out is MxK (all as long arrays)
/*
  mat_mat

  Computes the product of two matrices.

  Parameters
  ----------
  out : double*
    Storage for the resulting PRODUCT matrices.
  A : double*
  B : double*
    The two matrices to multiply.
  N : int
  M : int
  K : int
    The size of two matrices , `out`M*K, `A`M*N, and `B` N*K.

  Returns
  -------
  out : double*
    (Output by reference.) The PRODUCT of `A` and `B`.
*/
void mat_mat(double* out, double* A, double* B, int M, int N, int K);

#endif
