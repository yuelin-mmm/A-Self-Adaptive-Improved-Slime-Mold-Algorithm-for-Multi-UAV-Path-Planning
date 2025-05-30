'''
AI-SMA source code

paper:
A Self-Adaptive Improved Slime Mold Algorithm for Multi-UAV Path Planning
Yuelin Ma, Zeren Zhang, Meng Yao, Guoliang Fan
Drones
DOI: https://doi.org/10.3390/drones9030219
'''

import numpy

def transform_boundaries(lb, ub, dim):
    # Transform "low boundary" and "upper boundary" to arrays
    
    lb_array = numpy.array([lb for _ in range(dim)])
    ub_array = numpy.array([ub for _ in range(dim)])
    
    return lb_array, ub_array

def random_pop(pop_size, dim, lb_array, ub_array):
    # Randomly generate population positions
    
    pop = numpy.random.rand(pop_size, dim) * (ub_array - lb_array) + lb_array 

    return pop