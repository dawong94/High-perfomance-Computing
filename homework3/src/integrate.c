#include <omp.h>

double trapz_serial(double* fvals, double* x, int N)
{
    double h =0.0;
    double area = 0.0;

    for (int i=0;i<N-1;++i) {
        h = x[i+1]- x[i]; 
        area = area+ h*(fvals[i]+fvals[i+1])*0.5;
        h = 0.0;

    }


    return area;
}


double trapz_parallel(double* fvals, double* x, int N, int num_threads)
{
    int i;
    double area = 0.0;

    #pragma omp parallel for reduction(+:area) private(i) num_threads(num_threads)
     for (i=0;i<N-1;++i){
         area +=(x[i+1]- x[i])*(fvals[i]+fvals[i+1])*0.5;

     }

    return area;
}


double time_trapz_parallel(double* fvals, double* x, int N, int num_threads)
{
  double end, start = omp_get_wtime();
  trapz_parallel(fvals, x, N, num_threads);
  end = omp_get_wtime();
  return (end - start);
}


double simps_serial(double* fvals, double* x, int N)
{
    //int i;
    double area= 0.0;
    if (N %2 ==0){
      
      for (int i=0;i<N-3;i+=2) {
          area = area+ (x[i+2]-x[i])*(fvals[i]+4*fvals[i+1]+fvals[i+2])/6;
       }
      area = area + (x[N-1]- x[N-2])*(fvals[N-2]+fvals[N-1])*0.5;
    } else {
        for (int i=0;i<N-2;i+=2) {
          area = area+ (x[i+2]-x[i])*(fvals[i]+4*fvals[i+1]+fvals[i+2])/6;
        }
    }
    return area;
}


double simps_parallel(double* fvals, double* x, int N, int num_threads)
{
    int i;
    double area;
    if (N % 2 == 0) {
        area += (x[N-1] - x[N-2]) * ((fvals[N-1] + fvals[N-2])) * 0.5;
    }
    #pragma omp parallel num_threads(num_threads) private(i)
    {
        if (N % 2 == 1) {
            #pragma omp for reduction(+:area)
            for (i = 0; i < N-2; i+=2) {
                area += (x[i+2] - x[i]) * ((fvals[i] + 4*fvals[i+1] + fvals[i+2])) / 6;
            }
        } else {
            #pragma omp for reduction(+:area)
            
            for (i = 0; i < N-3; i+=2) {
                
                area += (x[i+2] - x[i]) * ((fvals[i] + 4*fvals[i+1] + fvals[i+2])) / 6;
            }
        }
    }
    return area;
}


double time_simps_parallel(double* fvals, double* x, int N, int num_threads,
                           int repeat)
{
  double end, start = omp_get_wtime();
  for (int i=0; i<repeat; ++i)
    simps_parallel(fvals, x, N, num_threads);
  end = omp_get_wtime();
  return (end - start) / (double)repeat;
}



double simps_parallel_chunked(double* fvals, double* x, int N,
                              int num_threads, int chunk_size)
{
    int i;
    double area;
    if (N % 2 == 0) {
        area += (x[N-1] - x[N-2]) * ((fvals[N-1] + fvals[N-2])) / 2;
    }
    #pragma omp parallel num_threads(num_threads) private(i)
    {
        if (N % 2 == 1) {
            #pragma omp for schedule(dynamic, chunk_size) reduction(+:area)
            for (i = 0; i < N-2; i+=2) {
                area += (x[i+2] - x[i]) * ((fvals[i] + 4*fvals[i+1] + fvals[i+2])) / 6;
            }
        } else {
            #pragma omp for schedule(dynamic, chunk_size) reduction(+:area)
            for (i = 0; i < N-3; i+=2) {
                area += (x[i+2] - x[i]) * ((fvals[i] + 4*fvals[i+1] + fvals[i+2])) / 6;
            }
        }
    }
    return area;
}


double time_simps_parallel_chunked(double* fvals, double* x, int N,
                                   int num_threads, int chunk_size,
                                   int repeat)
{
  double end, start = omp_get_wtime();
  for (int i=0; i<repeat; ++i)
    simps_parallel_chunked(fvals, x, N, num_threads, chunk_size);
  end = omp_get_wtime();
  return (end - start) / (double)repeat;
}
