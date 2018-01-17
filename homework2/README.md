# Homework #2

Homework #2 will test you on writing C code. In this assignment we will write
several basic linear algebra routines as well as some basic linear system
solvers. We will use these solvers in future assignments. This is the first
assignment which will test your code for performance. *This document is long but
please read carefully as it explains very thoroughly what to do and how to do
it.*

Repository layout:

* `homework2/`:

  Python wrapper of C library. The functions defined here allow you to use your
  C code from within Python. The test suite calls these functions which, in
  turn, call your C functions. **DO NOT MODIFY**. Any modifications will be
  ignored / overwritten when the homework is submitted.
  
  Please read the docstrings in this file to see how to call the wrapper and
  what kind of information is expected to be returned. Although you are not
  required to understand the source code of these wrappers it is, nonetheless,
  useful information.

* `include/`:

  The library headers containing (1) the C function prototypes and (2) the C
  function documentation. Document your C functions here.
  
* `lib/`:

  The compiled C code will be placed here as a shared object library named
  `libhomework2.so`.
  
* `report/`:

  Place your `report.pdf` here. See the *"Report"* section below.

* `src/`:

  C library source files. This is where you will provide the bodies of the
  corresponding functions requested below. *Do not change the way in which these
  functions are called.* Doing so will break the Python wrappers located in
  `homework2/wrappers.py`. Aside from writing your own tests and performing
  computations for your report, everything you need to write for this homework
  will be put in the files `src/linalg.c` and `src/solvers.c`.
  
* `ctests/`:

  Directory in which to place any optional C code used to debug and test your
  library as well as practice compiling and linking C code. See the file
  `ctests/example.c` for more information and on how to compile.
    
* `Makefile`: See *"Compiling and Testing"* below.

* `test_homework2.py`: A sample test suite for your own testing purposes.

## Compiling and Testing

The makefile for this homework assignment has been provided for you. It will
compile the source code located in `src/` and create a shared object library in
the directory `lib` named `libhomework2.so`.

Run,

```
$ make lib
```

to create `lib/libhomework2.so`. This library must exist in order for the Python
wrappers to work. As a shortcut, running

```
$ make test
```

will perform

```
$ make lib
$ python test_homework2.py
```

## Assignment

Provide the definitions for the following functions. The functions that return
by reference do so with a variable (in all cases, an array) named `out`. All
vectors are represented by arrays and all matrices are represented by long
arrays. That is, an *M*-by-*N* matirx *A* is represented by an array of length
*MN*.

**In `include/linalg.h` and `src/linalg.c`:**

* `vec_add`, `vec_sub`:

  Given arrays `v` and `w` of length `N` store the sum and difference,
  respectively, of these arrays in the array `out`.
  
* `vec_norm`:

  Given an array `v` of length `N` return the 2-norm / Euclidean norm of the
  vector, *|| v ||* by value. The two-norm of a vector is the square root of
  the dot product of `v` with itself. *(Hint: you may need a function from
  `math.h`.)*
  
* `mat_add`:

  Given two matrices `A` and `B` of size `M`-by-`N` store the sum of these
  matrices in a the long array `out`.
  
* `mat_vec`:

  Given a matrix `A` of size `M`-by-`N` and a vector `x` of length `N` store the
  matrix-vector product *Ax* in the array `out`. Note that the result will be a
  vector of length `M`.

* `mat_mat`:

  Given a matrix `A` of size `M`-by-`N` and a matrix `B` of size `N`-by-`K`
  store the matrix-matrix product *AB* in the array `out`. Note that the result
  will be a vector of length `MK`.

**In `include/solvers.h` and `src/solvers.c`:**

* `solve_lower_triangular`, `solve_upper_triangular`:

  Given an `N`-by-`N` lower triangular matrix `L` or upper triangular matrix
  `U`, respectively, and a vector `b` of length `N` solve the system *Lx = b* or
  *Ux = b*, respectively, storing the answer in the array `out`.
  
* `jacobi`, `gauss_seidel`:

  Given a strictly diagonally dominant `N`-by-`N` matrix `A` and a vector `b` of
  length `N` solve the system *Ax=b* using the Jacobi and Gauss-Seidel iterative
  methods, repsectively. The functions should store the result in the length `N`
  array `out`. These functions also accept a tolerance, `epsilon`, used for the
  convergence criterion. That is, the method should iterate until the 2-norm
  between successive approximations, *xkp1* and *xk*, is less than `epsilon`:
  
  ```c
  while (error > epsilon)
  {
    // (perform one iteration to compute xkp1 from xk)
    
    error = // 2-norm of the difference between successive approximations
  }
  ```
  
  **In addition**, the functions should *return by value* the number of
  iterations it takes to achieve this convergence criterion. The format of the
  return data via the Python wrapper is as follows:
  
  ```python
  from homework2 import jacobi
  x, num_iter = jacobi(A,b)
  ```
  
  (See the wrapper documentation for details.)

### 1) Tests - 60%

Your implementations will be run through the following test suite:

* (1pt) Does `vec_add` return the correct result on a variety of input?
* (1pt) Does `vec_sub` return the correct result on a variety of input?
* (1pt) Does `vec_norm` return the correct result on a variety of input?
* (1pt) Does `mat_add` return the correct result on a variety of input?
* (1pt) Does `mat_vec` return the correct result on a variety of input?
* (1pt) Does `mat_mat` return the correct result on a variety of input?
* (1pt) Does `solve_lower_triangular` return the correct result when `L` is diagonal
  and various choices of `b`? The output will be compared against the output of
  `numpy.linalg.solve`.
* (2pts) Does `solve_lower_triangular` return the correct result when `L` is an
  arbitrary lower-diagonal matrix and various choices of `b`? The output will be
  compared against the output of `numpy.linalg.solve`.
* (1pt) Does `solve_upper_triangular` return the correct result when `U` is diagonal
  an various choices of `b`? The output will be compared against the output of
  `numpy.linalg.solve`.
* (2pts) Does `solve_upper_triangular` return the correct result when `U` is an
  arbitrary lower-diagonal matrix and various choices of `b`? The output will be
  compared against the output of `numpy.linalg.solve`.

For the following tests, let `A` be an `n`-by-`n` 5-diagonal matrix consisting
of "5" along the main diagonal and "-1" on the (-2),(-1),(+1), and (+2)
off-diagonals and the vector `b` be the vector `[0, 1, 2, ..., n-1]`.
Additionally, use an initial guess (`x0`, in the previous hw) of all zeros.

* (3pts) Does `jacobi` return an approximate solution to the system *Ax = b* for
  various `n`?
* (3pts) Does `jacobi` return an approximate solution to the system *Ax = b* for
  various `epsilon` greater than `1e-15`?
* (3pts) Does `gauss_seidel` return an approximate solution to the system *Ax =
  b* for various `n`?
* (3pts) Does `gauss_seidel` return an approximate solution to the system *Ax = b* for
  various `epsilon` greater than `1e-15`?
* (3pts) Does `jacobi` return the expected number of iterations for this problem
  for various `n`? Your output will be compared directly with the output from a
  correct implementation.
* (3pts) Does `gauss_seidel` return the expected number of iterations for this
  problem for various `n`? Your output will be compared directly with the output
  from a correct implementation. *Note that you should expect around half as
  many iterations are required in `gauss_siedel` than in `jacobi`.*
  
**Total points:** 30/30

### 2) Report - 20%

Produce a PDF document named `report.pdf` in the directory `report` of the
repository. Please write your name and UW NetID (e.g. "cswiercz", *not* your
student ID number) at the top of the document. The report should contain
responses to the following questions:

1. (2pts) What is the computational complexity of `solve_lower_triangular` and
   `solve_upper_triangular` as a function of the matrix size `n`? That is, using
   big-oh notation list how much time / how many operations it takes to solve a
   triangular system if the input matrix *L* or *U* is *N*-by-*N*. Please read
   this article on
   [time complexity](https://en.wikipedia.org/wiki/Time_complexity) if you need
   a refresher on the topic. *Your response should contain an argument for your
   result!* Listing the complexity without any description will result in zero
   points.
   
2. (4pts) Copy the code you wrote for `solve_upper_triangular` and paste it into
   a markdown cell. You can format the cell to display code nicely using like
   so:
   
   <pre lang="no-highlight"><code>
   # The contents of In [1] formatted as "Markdown":
   
   The code I used for `solve_upper_triangular` is:
   
   ```c
   int gauss_seidel(double* out, ...)
   {
     // my code
   }
   ```
   
   (Explanation of my code.)
   
   </code></pre>
   
   See the example notebook `Scratch.nb` for a demonstrations. Answer the
   following questions about your implementation:
   
   * (2/4) Does this function have a contiguous memory access pattern?
   * (2/4) Assuming the cache create a copy of memory *at* and *after* a
     requested / particular location do you think `solve_upper_triangular` has
     any inefficiencies? Why or why not? (Note that the cache may not actually
     behave in this way but assume it does.)
 
3. (6pts) Just as in 2. above, copy the code you wrote for `gauss_seidel` and
   paste it into a markdown cell with nice code formatting. Answer the following
   questions about your implemenation:
   
   * (2/6)What are the memory requirements as a function of `n`, the system
     size? You do not need to calculate the number of bytes but at least give a
     description like "n-squared doubles for storing the matrix, n doubles for
     storing the right-hand side, etc."
   * (2/6) What parts of your code have contiguous access patterns?
   * (2/6) What parts do not have contiguous access patterns? Any thoughts on
     how to improve this, if possible?
   
**Total Points:** 12/12

### 3) Documentation - 10%

Provide documentation for the function prototypes listed in `include/linalg.h`
and `include/solvers.h` following the formatting described in the
[Grading document](https://github.com/uwhpsc-2016/syllabus/blob/master/Grading.md).
The format must match the following layout,

```c

  /*
      my_function
      
      (description of function)
      
      Parameters
      ----------
      arg1 : parameter type
          (Description of parameter)
      arg2 : parameter type,
          (Description of parameter)
          
      Returns
      -------
      argument or variable_name : output type
          (Description of output)
      arg2 : type, return by reference
          (Description of output)
  */
  double my_function(int arg1, double* arg2);
```

Write this documentation above the function prototype defined in the header
files, not in the source files. Example documentation for `vec_add` has already
been provided to you.

**Total Points:** 10/10 *(one for each function)*

### 4) Performance - 10%

Your implementation of `gauss_seidel` will be tested for performance. Failure to
produce a working implementation of `gauss_seidel` will result in a zero on the
performance evaluation. See the
[Grading document](https://github.com/uwhpsc-2016/syllabus/blob/master/Grading.md)
for more information on how the performance grading is done.
