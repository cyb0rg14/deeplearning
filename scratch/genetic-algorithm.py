"""
Genetic Algorithm: It is a stochastic method for function optimization inspired by the process of natural evolution - select parents to create children using the corssover and mutation processes.

The code is an implementation of the genetic algorithms for optimization. The algorithm is used to find the minmum value of a two-dimensional inverted Gaussian function centered at (7, 9). The algorithm consists of following steps:

- Initialize a population of binary bitstrings with random values
- Decode the binary bitstrings into numeric values, and evaluate the fitness (the objective function) for each individual in the population.
- Select the best individuals from the population using tournament selection based on the fitness.
- Create new offsprings from the selected individuals using the crossover operation.
- Apply the mutation operation on the offsprings to maintain diversity in the population.
- Repeat steps 2 to 5 until a stopping criterion is met.

The implementation includes functions for decoding, selection, crossover and mutation.
"""

import math
from numpy.random import rand, randint

# 2-D inverted GF -> centered at (7, 9)
def objective(x):
    y = math.exp(((x[0] - 7) ** 2) + (x[1] - 9) ** 2)
    return y

# decodes binary bitstrings -> numbers & scales the values
def decode(bounds, n_bits, bitstring):
    decoded = [] 
    largest = 2 ** n_bits
    for i in range(len(bounds)):
        start, end = i * n_bits, (i * n_bits) + n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
    return decoded

# select the next individual for the next generation based on their fitness score
def selection(population, scores, k=3):
    selection_ix = randint(len(population))
    for ix in randint(0, len(population), k-1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]





















