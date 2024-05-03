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

