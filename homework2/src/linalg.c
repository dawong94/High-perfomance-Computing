#include "linalg.h"
#include <math.h>

// EXAMPLE *OPTIONAL* FUNCTION DOCS
//
// (only write about tricky implementation details here)
/*
  vec_add

  We use the advanced method of computing the sum of two elements of an array
  and storing the result in a third array. It's all very complex so I must talk
  about it here.
*/
void vec_add(double* out, double* v, double* w, int N)
{
  for (int i=0; i<N; ++i)
    {
      out[i] = v[i] + w[i];
    }
}

void vec_sub(double* out, double* v, double* w, int N)
{
  for (int j=0; j<N; ++j)
    {
      out[j] = v[j] - w[j];
    }
}

double vec_norm(double* v, int N)
{
  double sum =0.0;
  for (int k=0;k<N;k++)
    {

      sum += v[k]*v[k];
    }

  return sqrt(sum);
}

// represent out, A, and B by arrays of length M*N
void mat_add(double* out, double* A, double* B, int M, int N)
{
  for (int i=0;i<M*N;++i)
    {
     // for (int j=0;j<N;++j)
       // {
          out[i] = A[i]+B[i];
       // }  
    }
}

// represent A by an array of length M*N
void mat_vec(double* out, double* A, double* x, int M, int N)
{
  for (int i=0; i<M;++i) 
  {
      for (int j=0; j<N;++j)
      {
          out[i]+= A[i*N+j]*x[j];
      }
  }

}

// A is MxN, B is NxK, and out is MxK (all as long arrays)
void mat_mat(double* out, double* A, double* B, int M, int N, int K)
{
    for (int i=0; i<M; ++i)
      for (int k=0; k<N; ++k)  // swapped
        for (int j=0; j<K; ++j)  // swapped
          out[i*K + j] += A[i*N + k] * B[k*K + j];
}
