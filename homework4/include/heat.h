#ifndef __homework_4_heat_h
#define __homework_4_heat_h

#include <mpi.h>

/*
  heat_serial

  A serial version of solving the periodic heat equation on a uniform grid. Use
  for comparison with the MPI distributed version.

  The forward Euler method for numerical solving PDEs is unstable for large
  (even only moderately) values of `dt`. Make sure `dt` is small. Examples are
  given in the Python code that calls this function.

  Parameters
  ----------
  u : double*
      Input data array. Represents the heat / temperature across a 1d rod.
      Assumes that the data points are equally spaced in the space domain.
  Nx : size_t
      The length of `u`.
  dt : double
      The size of the time step to take with each iteration.
  Nt : size_t
      Number of time-steps to perform.
  dx : double
      The change of position.    

  Returns
  -------
  None
      The array `u` is modified in place.

*/
void heat_serial(double* u, double dx, size_t Nx, double dt, size_t Nt);

/*
  heat_serial

  A parallel version of solving the periodic heat equation on a uniform grid. Use
  for comparison with the MPI distributed version.

  The forward Euler method for numerical solving PDEs is unstable for large
  (even only moderately) values of `dt`. Make sure `dt` is small. Examples are
  given in the Python code that calls this function.

  Parameters
  ----------
  uk : double*
      chunked Input data array. Represents the heat / temperature across a 1d rod.
      Assumes that the data points are equally spaced in the space domain.
  dx : double
      The change of position.
  dt : double
      The size of the time step to take with each iteration.
  Nt : size_t
      Number of time-steps to perform.
  Nx : size_t
      number of element in a chunk
  comm :  MPI_Comm
         communicator 
  Returns
  -------
  None
      The array `uk` is modified in place.

*/
void heat_parallel(double* uk, double dx, size_t Nx, double dt, size_t Nt,
                   MPI_Comm comm);

#endif
