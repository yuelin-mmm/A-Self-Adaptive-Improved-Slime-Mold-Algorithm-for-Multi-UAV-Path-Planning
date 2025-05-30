'''
AI-SMA source code

paper:
A Self-Adaptive Improved Slime Mold Algorithm for Multi-UAV Path Planning
Yuelin Ma, Zeren Zhang, Meng Yao, Guoliang Fan
Drones
DOI: https://doi.org/10.3390/drones9030219
'''

import numpy

def fitness_evaluation(objf, pop):
    # Calculate fitness
    
    all_fitness = numpy.asarray([objf(ind) for ind in pop]) # All individuals fitness
    sorted_indices = numpy.argsort(all_fitness) # The sorted index
    sort_positions = pop[sorted_indices] # The sorted position
    sort_fitness = all_fitness[sorted_indices] # The sorted fitness
    
    current_best_position = sort_positions[0] 
    current_best_fitness = sort_fitness[0] 
    current_worst_fitness = sort_fitness[-1]

    return sort_positions, sort_fitness, \
        current_best_fitness, current_best_position, current_worst_fitness
