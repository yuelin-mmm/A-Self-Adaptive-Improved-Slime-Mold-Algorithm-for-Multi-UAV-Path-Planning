'''
AI-SMA source code

paper:
A Self-Adaptive Improved Slime Mold Algorithm for Multi-UAV Path Planning
Yuelin Ma, Zeren Zhang, Meng Yao, Guoliang Fan
Drones
DOI: https://doi.org/10.3390/drones9030219
'''

import numpy
import Initialization
import RankingDE
import Weight
import Evaluation

def AISMA(objf, lb, ub, dim, pop_size, max_iter, cr, dr):
    
    #Initialization    
    Convergence_curve=[] 
    Convergence_position = [] 
    N_count= 0 # Stagnation counting
    weight = numpy.zeros((pop_size, dim))
    
    lb_array, ub_array = Initialization.transform_boundaries(lb, ub, dim)
    pop = Initialization.random_pop(pop_size, dim, lb_array, ub_array) # generate population
    sort_positions, sort_fitness, current_best_fitness, \
        current_best_position, current_worst_fitness = Evaluation.fitness_evaluation(objf, pop) # evaluate 
    current_best_worst = current_best_fitness - current_worst_fitness
    weight = Weight.update_weight(weight, current_best_fitness, current_best_worst, \
        pop_size, sort_fitness, dim) # update weight
    Probabilities = RankingDE.cal_probability(pop_size) 

    global_fitness = numpy.inf 
    global_position = current_best_position 
    last_best_fitness = 0 # best fitness in the last iteration
    
    for it in range(max_iter):
        # main iteration
        
        z = numpy.log10(1 + 0.8 * numpy.exp((it+1)/max_iter))
        a = numpy.arctanh(-((it+1) / max_iter) + 1) 

        for i in range(pop_size):
            
            vb = numpy.random.uniform(-a, a, (1, dim))
            mutant = RankingDE.cal_mutant(pop_size, Probabilities, i, sort_positions, vb)
            
            #case1 - random
            new_pop1 = Initialization.random_pop(pop_size, dim, lb_array, ub_array)
            #case2 - exploration & exploitation
            new_pop2 = numpy.full(dim, 0.0) #initialize
            
            for d in range(0, dim):

                rand1 = numpy.random.rand()
                # random individual index
                A = numpy.random.randint(1, pop_size+1) - 1
                B = numpy.random.randint(1, pop_size+1) - 1
                
                if(rand1 < cr or i == d):
                    #exploitation
                    new_pop2[d] = global_position[d] + vb[0][d] * (weight[i,d] * sort_positions[A,d] - sort_positions[B,d])    
                else:
                    #exploration 
                    new_pop2[d] = mutant[0][d]

            # random or not
            rand2 = numpy.random.rand()        
            if(rand2 < z):
                pop_temp = new_pop1[0]
            else:
                pop_temp = new_pop2

            # greedy selection - new or old
            if(objf(pop_temp) < sort_fitness[i]):
                sort_positions[i] = pop_temp
        
        # evaluate 
        sort_positions, sort_fitness, current_best_fitness, \
            current_best_position, current_worst_fitness = Evaluation.fitness_evaluation(objf, sort_positions)
        current_best_worst = current_best_fitness - current_worst_fitness
        weight = Weight.update_weight(weight, current_best_fitness, current_best_worst, \
            pop_size, sort_fitness, dim)
        
        if(abs(current_best_fitness - last_best_fitness) < 0.01):
            # stagnation
            N_count += 1
        else:
            N_count = 0

        last_best_fitness = current_best_fitness
        
        # update global fitness & global position
        if(current_best_fitness < global_fitness):
            global_fitness = current_best_fitness
            global_position = current_best_position

        # perturbation
        if(N_count >= 15):

            for k in range(int(pop_size*(1-dr)), pop_size):
                rand4 = numpy.random.rand()
                pop_r = Initialization.random_pop(1, dim, lb_array, ub_array)
                sort_positions[k] =  rand4 * global_position + (1 - rand4) * pop_r
                sort_fitness[k] = objf(sort_positions[k])

            # evaluate
            sort_positions, sort_fitness, current_best_fitness, \
                current_best_position, current_worst_fitness = Evaluation.fitness_evaluation(objf, sort_positions)
            current_best_worst = current_best_fitness - current_worst_fitness
            weight = Weight.update_weight(weight, current_best_fitness, current_best_worst, \
                pop_size, sort_fitness, dim)           
            
            last_best_fitness = current_best_fitness 
            
            # update global fitness & global position
            if(current_best_fitness < global_fitness):
                global_fitness = current_best_fitness
                global_position = current_best_position
                N_count = 0
    
        Convergence_curve.append(float(global_fitness)) 
        Convergence_position.append(global_position)
        
    return Convergence_curve, Convergence_position
