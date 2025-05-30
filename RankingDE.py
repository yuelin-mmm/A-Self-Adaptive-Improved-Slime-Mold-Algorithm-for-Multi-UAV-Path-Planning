'''
AI-SMA source code

paper:
A Self-Adaptive Improved Slime Mold Algorithm for Multi-UAV Path Planning
Yuelin Ma, Zeren Zhang, Meng Yao, Guoliang Fan
Drones
DOI: https://doi.org/10.3390/drones9030219
'''

import numpy

def cal_probability(pop_size):
    # Calculate the probability that each individual is selected after sorting
    
    Probabilities = []
    for j in range(1, pop_size+1):
        Probabilities.append((pop_size - j) / pop_size)
    
    return Probabilities

def cal_mutant(pop_size, Probabilities, i, sort_positions, vb):
    # Calculate the mutation operator
    
    r1 = numpy.random.randint(1, pop_size+1) #r1
    while(numpy.random.uniform() > Probabilities[r1-1] or r1 == i):
        r1 = numpy.random.randint(1, pop_size+1)   

    r2 = numpy.random.randint(1, pop_size+1) #r2
    while(numpy.random.uniform() > Probabilities[r2-1] or r2 == r1 or r2 == i):
        r2 = numpy.random.randint(1, pop_size+1)

    r3 = numpy.random.randint(1, pop_size+1) #r3
    while(r3 == r2 or r3 == r1 or r3 == i):
        r3 = numpy.random.randint(1, pop_size+1)

    x_a = sort_positions[r1-1]
    x_b = sort_positions[r2-1]
    x_c = sort_positions[r3-1]
    
    # DE/rand/1
    mutant = x_a + 2 * numpy.random.random() * vb * (x_b - x_c)

    return mutant