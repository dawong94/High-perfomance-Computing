#include "linalg.h"
#include "solvers.h"
#include <stdlib.h>
// matrices L, U, and A are all long arrays of size NxN
// b is an array of length N

void solve_lower_triangular(double* out, double* L, double* b, int N)
{
  out[0]=b[0]/L[0];
  for (int i=1;i<N;++i){
      out[i] = b[i];
      for (int j=i-1;j>=0;j--){
          out[i] -= L[i*N + j]*out[j];
      }
      out[i] = out[i] / L[i*N + i];
  }  
}

void solve_upper_triangular(double* out, double* U, double* b, int N)
{
  out[N-1] = b[N-1]/U[N*N -1];

  for (int i=(N-2); i>=0; i--){

      out[i] = b[i];
      for (int j=i+1; j<N; j++)
      {
          out[i] -= U[i*N + j]*out[j];
      }
      out[i] = out[i] / U[i*N + i];
  }
}

void decompose(double* D, double* L, double* U, double* A, int N)
{
    for (int i=0;i<N;++i)
    {
        D[i*N+i] = A[i*N +i];
    }
    //get lower triangular
    for (int k=1;k < N;++k)

    {
        for (int v=0;v < k;++v)
        {
            L[k*N+v] = A[k*N + v];
        }
    }

    for (int c = N-2; c>=0; --c)
    {
        for (int w = N-1; w>c;--w)
        {
            U[c*N + w] = A[c*N + w];
        }
    }
}

void copy(double* xk,double* out,int N)
{


    for (int i=0; i<N;++i){
        xk[i] = out[i];
    }
}

//void jacobi_step(double* out, double* D,double* L,double* U, int N, double* xk, double* b)
void jacobi_step(double* out, double* D,double* T,double* K, double* P, int N, double* xk, double* b)
{    //jacobi_steo(out,D,T,K,P,N,xk,b);
   // double* T = (double*) malloc(N*N * sizeof(double));//(L+U)
    //double* K = (double*) malloc(N * sizeof(double));//dot(T,xk)
   // double* P = (double*) malloc(N * sizeof(double));//b-dot(T,xk)
    //mat_add(T,L,U,N,N);//get the T
    mat_vec(K,T,xk,N,N);//dot(T,xk)
    vec_sub(P,b,K,N);//b-dot(T,xk)
    for (int i=0;i<N;++i)
    {
        out[i]= P[i]/D[i*N + i];
    }
   // free(T);
   //free(K);
   // free(P);
}


int jacobi(double* out, double* A, double* b, int N, double epsilon)
{
  double* D = (double*) malloc(N*N * sizeof(double));
  double* L = (double*) malloc(N*N * sizeof(double));
  double* U = (double*) malloc(N*N * sizeof(double));
  double* T = (double*) malloc(N*N * sizeof(double));//(L+U)
  double* K = (double*) malloc(N * sizeof(double));//dot(T,xk)
  double* P = (double*) malloc(N * sizeof(double));//b-dot(T,xk)
  double* error = (double*) malloc(N * sizeof(double));//out-xk
  double* xk = (double*) malloc(N * sizeof(double));  
  decompose(D, L, U, A, N);// call decompose method
  mat_add(T,L,U,N,N);//get the T
  int itr =1;
  for (int i =0; i<N;++i) {
      xk[i] =0;
  }
  
  //jacobi_step(out,D,L,U,N,xk,b);//get the first xkp1
  jacobi_step(out,D,T,K,P,N,xk,b); 
  vec_sub(error,out,xk,N);//out-xk
 
  while (vec_norm(error,N) > epsilon)
  {
        
        
        K = (double*) malloc(N * sizeof(double));//dot(T,xk)
        P = (double*) malloc(N * sizeof(double));//b-dot(T,xk)
        copy(xk,out,N);//xk=xk1
        //jacobi_step(out,D,L,U,N,xk,b);
        jacobi_step(out,D,T,K,P,N,xk,b);
        vec_sub(error,out,xk,N);
        ++itr;
  
  }

  free(D);
  free(L);
  free(U);
  free(xk);
  free(error);
  free(T);
  free(K);
  free(P);  
  return itr;
}

void gauss_step(double* out, double* S,double* K, double* P, int N, double* xk, double* b, double* L)
{
    mat_vec(K,L,xk,N,N);//dot(L,xk)
    vec_sub(P,b,K,N);//b-dot(L,xk)
    solve_upper_triangular(out,S,P,N);
}


int gauss_seidel(double* out, double* A, double* b, int N, double epsilon)
{
  double* D = (double*) malloc(N*N * sizeof(double));
  double* L = (double*) malloc(N*N * sizeof(double));
  double* U = (double*) malloc(N*N * sizeof(double));
  double* S = (double*) malloc(N*N * sizeof(double));//(D+U) 
  double* error = (double*) malloc(N * sizeof(double));//out-xk
  double* xk = (double*) malloc(N * sizeof(double));
  double* K = (double*) malloc(N * sizeof(double));//dot(L,xk)
  double* P = (double*) malloc(N * sizeof(double));//b-dot(L,xk)
  decompose(D, L, U, A, N);// call decompose method
  mat_add(S,D,U,N,N);//get the S 
  int itr = 1;  
  gauss_step(out,S,K,P,N,xk,b,L);
  vec_sub(error,out,xk,N);
  while (vec_norm(error,N) > epsilon)
  {
        
        K= (double*) malloc(N * sizeof(double));//dot(T,xk)
        P= (double*) malloc(N * sizeof(double));//b-dot(T,xk)
        copy(xk,out,N);//xk=xk1
        gauss_step(out,S,K,P,N,xk,b,L);
        vec_sub(error,out,xk,N);
        ++itr;
  }
   
  free(D);
  free(L);
  free(U);
  free(xk);
  free(error);
  free(S);
  free(K);
  free(P);  
  return itr;
}


