"""
Genetic algorithms are a type of optimization algorithm inspired by the process of natural selection and evolution. They work by evolving a population of candidate solutions over several generationsto find the best solution to a problem.

Here's a concise breakdown:
- Initialization: Create a population of random candidate solutions (chromosomes).
- Selection: Evaluate the fitness of each chromosome and select the fittest ones to reproduce.
- Crossover: Create new offspring by combining the genetic material (genes) of selected parents.
- Mutation: Introduce random changes in the offspring to maintain genetic diversity.
- Replacement: Replace the least fit members of the population with the new offspring.
- Termination: Repeat the process until a termination condition is met (e.g., a maximum number of generations is reached or an acceptable solution is found).
"""

import random
import numpy as np

# X Genes -> One chromosome | X chromosomes -> One population
POPULATION_SIZE = 10 # -> it represents 10 no of chromsomes
GENE_SIZE = 8 # -> each chromosome would have 8 genes

population = [
    [random.randint(0, 1) for _ in range(GENE_SIZE)]
    for _ in range(POPULATION_SIZE)
]
# print(population)

""" output:
[[1, 1, 0, 1, 1, 0, 1, 0],
 [1, 1, 1, 1, 1, 1, 0, 1],
 [1, 1, 1, 1, 0, 1, 1, 0],
 [0, 0, 0, 1, 0, 0, 1, 0],
 [1, 1, 0, 0, 1, 0, 0, 0],
 [0, 0, 0, 1, 1, 0, 0, 0],
 [0, 1, 0, 1, 1, 1, 1, 0],
 [0, 1, 0, 1, 1, 0, 0, 1],
 [1, 1, 1, 0, 1, 1, 0, 0],
 [1, 1, 0, 1, 1, 0, 0, 1]]
"""


def fitness(chromsome):
    THRESHOLD = 200
    # convert binary to integer
    chromsome = int("".join(map(str, chromsome)), 2) 
    # [0, 0, 0, 1, 1, 1, 1, 1] -> 31 (after this step)
    fitness_score = 255 - abs(THRESHOLD - chromsome)
    # 255 - abs(200 - 31) => 255 - 169
    return fitness_score
     
# print(fitness(population[0])) # -> 86

def crossover(parent1, parent2):
    # choose a random point b/w range(1, GENE_SIZE)
    crossover_point = random.randint(1, len(parent1))    
    # new child would have genes from both parent1, parent2
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# parent1, parent2 = population[0], population[1]
# child1, child2 = crossover(parent1, parent2)
# print(parent1, parent2)
# print(child1, child2)
#
""" Output:
[0, 1, 1, 1, 0, 1, 1, 1] [1, 0, 0, 1, 0, 1, 0, 0] -> parents
[1, 1, 1, 1, 0, 0, 1, 0] [0, 1, 1, 1, 0, 1, 0, 0] -> childs
"""


def mutate(chromosome):
    # simple mutatation: reverse the gene with a 10% chance
    for i in range(len(chromosome)):
        if random.random() < 0.10:
            if chromosome[i]  == 0: chromosome[i] = 1
            elif chromosome[i] == 1: chromosome[i] = 0
    return chromosome

# chromosome = population[0]
# print(chromosome, mutate(chromosome))

# [1, 0, 1, 1, 0, 0, 0, 1] -> [1, 0, 1, 1, 0, 1, 0, 0]

"""
- We select 2 best chromosome as parents.
- And replace rest of the 8 chromosomes with the
childs (crossover of those two parents).
- So the new population would be:
    population = parents + offsprings
    (10)       =   (2)     +   (8)
"""

num_generations = 10
for generation in range(num_generations):
    # Evaluate fitness 
    fitness_scores = [fitness(chromsome) for chromsome in population]
    
    # Selection    
    parents = [population[i] for i in np.argsort(fitness_scores)[-2:]]

    # Crossover
    offsprings = []
    for i in range(POPULATION_SIZE - len(parents)):
        child = crossover(parents[0], parents[1])
        offsprings.append(child)

    # Mutation 
    for i in range(len(offsprings)):
        offsprings[i] = mutate(offsprings[i])

    # Replacement
    population = parents + offsprings

    # Display the best chromsome 
    best_chromosome = max(population, key=fitness)
    print("Chromsome:", best_chromosome,"fitness:", fitness(best_chromosome)) 
          
