'''
AI-SMA source code

paper:
A Self-Adaptive Improved Slime Mold Algorithm for Multi-UAV Path Planning
Yuelin Ma, Zeren Zhang, Meng Yao, Guoliang Fan
Drones
DOI: https://doi.org/10.3390/drones9030219
'''

import numpy

def update_weight(weight, current_best_fitness, current_best_worst, \
    pop_size, sort_fitness, dim):
    # Update weight
    
    eps = numpy.finfo(float).eps # a small constant
    best_worst = current_best_worst + eps
    
    for i in range(pop_size):
        if(i <= pop_size/2):
            weight[i] = 1 + numpy.random.rand() * numpy.log10(
            ((current_best_fitness - numpy.repeat(sort_fitness[i], dim)) / best_worst) + 1)
        else:
            weight[i] = 1 - numpy.random.rand() * numpy.log10(
            ((current_best_fitness - numpy.repeat(sort_fitness[i], dim)) / best_worst) + 1)
            
    return weight
