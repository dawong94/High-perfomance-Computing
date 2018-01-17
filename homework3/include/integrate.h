
/*
  integrate.h
  -----------

  Defines routines for numerically integrating (x_i,fval_i) data. That is, given
  a function f, some points along a domain x = [x_0, x_1, ..., x_{N-1}] and
  function values

  fvals = [f(x_0), f(x_1), ..., f(x_{N-1})]

  numerically approximate the integral of f using these function evaluations.
  These are meant to replecate the work done by `scipy.integrate.trapz` and
  `scipy.integrate.simps`, but written in C and using OpenMP.
*/


/*  returns numerically approximate the integral of f using given parameters
    by using trapz rule
    ----------
    x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    
    Returns
    -------
    result : double
         returns numerically approximate integral of f 
        
*/
double trapz_serial(double* fvals, double* x, int N);

/*  returns numerically approximate the integral of f using given parameters by using OpenMp 
    parallel tools and trapz rule
    ----------
    x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    num_threads : int
        The number of threads are required
    Returns
    -------
    result : double
         returns numerically approximate integral of f 
        
*/
double trapz_parallel(double* fvals, double* x, int N, int num_threads);

/*  returns running time of this method using given domain  and function valuse fvals
    ----------
     x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    num_threads : int
        The number of threads are required
    repeat : int
        the number of repeat times
    Returns
    -------
    result : double
        running time of traps_parallel
*/

double time_trapz_parallel(double* fvals, double* x, int N, int num_threads);

/*  returns numerically approximate the integral of f using given parameters by using
    simpson rule.
    ----------
    x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    
    Returns
    -------
    result : double
         returns numerically approximate integral of f */
double simps_serial(double* fvals, double* x, int N);

/*  returns numerically approximate the integral of f using given parameters by using OpenMp 
    parallel tools and simpson rule
    ----------
    x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    num_threads : int
        The number of threads are required
    Returns
    -------
    result : double
         returns numerically approximate integral of f 
        
*/
double simps_parallel(double* fvals, double* x, int N, int num_threads);

/*  returns running time of this method using given domain  and function valuse fvals
    ----------
     x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    num_threads : int
        The number of threads are required
    repeat : int
        the number of repeat times
    Returns
    -------
    result : double
        running time of traps_parallel
*/
double time_simps_parallel(double* fvals, double* x, int N, int num_threads,
                           int repeat);

/*  returns numerically approximate the integral of f using given parameters by using OpenMp 
    looping chunking techniques and simpson rule
    ----------
    x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    num_threads : int
        The number of threads are required
        
    chunk_size: int
        loop chunking size
    Returns
    -------
    result : double
         returns numerically approximate integral of f 
        
*/
double simps_parallel_chunked(double* fvals, double* x, int N, int num_threads,
                              int chunk_size);

/*  returns running time of this method using given domain  and function valuse fvals
    ----------
     x : double*
         x
    fvals : double*
         f values
    N : int
        the length of array
    num_threads : int
        The number of threads are required
    repeat : int
        the number of repeat times
    chunk_size: int
        loop chunking size
    Returns
    -------
    result : double
        running time of traps_parallel
*/
double time_simps_parallel_chunked(double* fvals, double* x, int N,
                                   int num_threads, int chunk_size, int repeat);
