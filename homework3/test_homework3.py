import unittest
import numpy

from numpy import array, linspace, pi, sin, exp
from scipy.integrate import trapz, simps
import time

from homework3 import (
    trapz_serial,
    trapz_parallel,
    time_trapz_parallel,
    simps_serial,
    simps_parallel,
    time_simps_parallel,
    simps_parallel_chunked,
    time_simps_parallel_chunked,
)

#
# Implement your tests here.
#
class TestIntegrate(unittest.TestCase):
    def test_trap_serial(self):
        t_start =time.time();
        x = linspace(-1, 3, 1e7)
        y = sin(exp(x))
        #x= linspace(-1,pi,8)
        #y=sin(x)
        area = trapz_serial(y,x)
        actual = trapz(y,x)
        print 'time: ', (time.time()- t_start)
        print area,actual
        
    def test_trap_parallel(self):
        
        x = linspace(-1, 3, 1e7)
        y = sin(exp(x))
        #x= linspace(-1,pi,8)
        #y=sin(x)
        area = trapz_parallel(y,x)
        actual = trapz(y,x)
        print 'time: ', time_trapz_parallel(y,x,num_threads=4)
        print area,actual

     
    def test_simps_serial(self):
        time_start = time.time()
        x = linspace(-1, 3, 1e7)
        y = sin(exp(x))
        area = simps_serial(y,x)
        actual = simps(y,x)
        print area
        print actual
        print 'time: ', str(time.time()-time_start)
    # serial
    
    def test_simps_parallel(self):
        x = linspace(-1, 3, 1e7)
        y = sin(exp(x))
        area = simps(y,x)
        real = simps_parallel(y,x,4)
        print 'time: ', time_simps_parallel(y,x,num_threads=4)
        print area
        print real
        
    def test_simps_parallel_chunk(self):
        
        x = linspace(-1, 3, 1e7)
        y = sin(exp(x))
        area = simps(y,x)
        real = simps_parallel_chunked(y,x,4,200)
        print 'time: ', time_simps_parallel_chunked(y,x,num_threads = 4,chunk_size=1000)
        print area 
        print real
    
    def null_test(self):
        self.assertTrue(False)


if __name__ == '__main__':
    #
    # Example timing code: these are on trapz
    #
    print '\n===== Running Timings ====='
    x = linspace(-1, 3, 1e7)
    y = sin(exp(x))

    t1 = time_trapz_parallel(y, x, num_threads=1)
    print 'trapz threads = %d, time = %f'%(1, t1)

    t2 = time_trapz_parallel(y, x, num_threads=2)
    print 'trapz threads = %d, time = %f'%(2, t2)

    t4 = time_trapz_parallel(y, x, num_threads=4)
    print 'trapz threads = %d, time = %f'%(4, t4)
       
    st1 = time_simps_parallel(y, x, num_threads=1)
    print 'simps threads = %d, time = %f'%(1, st1)

    st2 = time_simps_parallel(y, x, num_threads=2)
    print 'simps threads = %d, time = %f'%(2, st2)

    st4 = time_simps_parallel(y, x, num_threads=4)
    print 'simps threads = %d, time = %f'%(4, st4)
    
    cst1 = time_simps_parallel(y, x, num_threads=1)
    print 'simps chunked = %d, time = %f'%(1, cst1)

    cst2 = time_simps_parallel(y, x, num_threads=2)
    print 'simps chunked = %d, time = %f'%(2, cst2)

    cst4 = time_simps_parallel(y, x, num_threads=4)
    print 'simps chunked = %d, time = %f'%(4, cst4)

    #
    # Run the test suite you created above
    #
    print '\n===== Running Tests ====='
    unittest.main(verbosity=2)

