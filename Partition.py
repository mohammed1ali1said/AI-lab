import math
import numpy as np
import Objects as objs


# CROWDING
def calculate_boltzmann_probability(fitness1, fitness2, T):
    # Calculate the fitness difference
    delta_fitness = fitness1 - fitness2

    # Calculate the Boltzmann replacement probability
    probability = math.exp(-delta_fitness / T) / (1 + math.exp(-delta_fitness / T))

    return probability



# FITNESS SHARING
def sharing_function(distance, sigma, alpha=1):

    if distance < sigma:
        return 1 - (distance / sigma) ** alpha
    else:
        return 0


def adjust_fitness_with_sharing(population, raw_fitness, sigma):

    adjusted_fitness = []
    for i in range(len(population)):
        niche_sum = sum(sharing_function(calculate_distance_sudoku(population[i], population[j]), sigma) for j in range(len(population)))
        adjusted_fitness.append(raw_fitness[i] / niche_sum if niche_sum > 0 else raw_fitness[i])
    return adjusted_fitness
def calculate_distance_sudoku(grid1,grid2): # takes 2 sudoku grids and calculates the distance between them (or how similar they are)
    # Ensure the grids are numpy arrays
    grid1 = np.array(grid1)
    grid2 = np.array(grid2)

    # Calculate the number of differing positions
    distance = np.sum(grid1 != grid2)

    return distance



def partition_to_niches_sudoku(population: list[objs.SudokuIndividual], sigma):

    niches: list[objs.SudokuIndividual] = []

    for indiv in population:
        placed = False
        for niche in niches:
            # Check if the grid fits into any existing niche
            if any(calculate_distance_sudoku(indiv.grid, niche_member.grid) <= sigma for niche_member in niche):
                niche.append(indiv)
                placed = True
                break
        if not placed:
            # Create a new niche if the grid does not fit into any existing niche
            niches.append([indiv])

    return niches

def calculate_distance_binpack():
    pass



# SPECIATION
