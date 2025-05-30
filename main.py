from mealpy import FloatVar
import AISMA
import numpy
import opfunu

dim = 30 # dimension
    
def run_AISMA(objf, lb, ub, dim, pop_size, max_iter, cr, dr):
    global_best, global_postion = AISMA.AISMA(objf, lb, ub, dim, pop_size, max_iter, cr, dr)
    return global_best, global_postion


def cec2017_fobj1(x):
    # CEC 2017 Function 1
    
    funcs = opfunu.get_functions_based_classname('F12017')
    func = funcs[0](ndim = dim) 
    F = func.evaluate(x)
    return F


if __name__ == '__main__':
    
    max_iteration = 1000 # maximum iteration
    pop_size = 30 # population size
    cr = 0.5 # Crossover Rate
    dr = 0.1 # Perturbation Rate
    lb = -100
    ub = 100
    
    i = 1 # CEC2017 function index
    
    problem = {
        "obj_func": cec2017_fobj1,
        "bounds": FloatVar(lb=(lb, )*dim, ub=(ub, )*dim),
        "minmax": "min",
        "log_to": None # do not pring log file
    }

    with open("AI-SMA.txt", 'a') as f:
        f.write("Function" + str(i) + "\n")

    global_best, global_postion = run_AISMA(cec2017_fobj1, lb, ub, dim, pop_size, max_iteration, cr, dr)
    global_best_log = [numpy.log(x) for x in global_best]

    with open("AI-SMA.txt", 'a') as f:
        f.write(str(global_best[-1]) + "\n") # global best fitness 
        f.write(str(global_best) + "\n") # global best convergence curve
        f.write(str(global_best_log) + "\n") # global best convergence curve(LOG)
    
    