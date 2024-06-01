import numpy as np
import argparse
import random
import sys
import time
import math
import matplotlib.pyplot as plt
import crossOverMethods as com
import Mutations as mut
import parentSelection
import parentSelection as ps
import  Objects as objects
import copy

# STRINGS ORIGINAL FITNESS

def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score

# STRINGS BULLS COWS FITNESS

def evaluation(genes):
    word = "Hello, world!"
    dict = {}  # word character frequencies
    geneCount = {}  #  genes character frequences
    for i in range(len(word)):  # count "word" character frequencies
        if word[i] in dict:
            dict[word[i]] += 1
        else:
            dict[word[i]] = 1

    bulls = 0
    for i in range(len(word)):  # calculate bulls
        if word[i] == genes[i]:
            bulls += 1

    for i in range(len(genes)):  # count "genes" character frequencies
        if genes[i] in geneCount:
            geneCount[genes[i]] += 1
        else:
            geneCount[genes[i]] = 1
    cows = 0
    for i in geneCount:  # calculate the total matches, including bulls and cows
        if i in dict:
            cows += min(geneCount[i], dict[i])
    cows -= bulls  # since the cows include bulls, lets substract the bulls

    bull_weight = 1
    cow_weight = 0.2
    score = bull_weight * bulls + cow_weight * cows
    return score



# SUDOKU FITNESS


def calc_fitness_sudoku_general(sudoku_grid):

    size = len(sudoku_grid)
    #grid_len = len(sudoku_grid)
    # for row in sudoku_grid:
    #     print(row)
    row_score_sum = 0
    col_score_sum = 0
    box_score_sum = 0

    for i in range(0,len(sudoku_grid)):
        row_score_sum += calc_row_fitness(sudoku_grid,i)

    for i in range(0,len(sudoku_grid)):
        col_score_sum += calc_col_fitness(sudoku_grid,i)

    N = len(sudoku_grid)
    step_size = int(math.sqrt(size))

    for i in range(0, N, step_size):
        for j in range(0, N, step_size):
            box_score_sum += calc_box_fitness(sudoku_grid,i,j,int(math.sqrt(size)))

    # print("row sum: ",row_score_sum)
    # print("column sum: ",col_score_sum)
    # print("boxes sum: ", box_score_sum)
    return row_score_sum+col_score_sum+box_score_sum


def calc_fitness_sudoku(sudoku_individual: objects.SudokuIndividual):
    sudoku_grid = sudoku_individual.grid
    size = len(sudoku_grid)
    #grid_len = len(sudoku_grid)
    # for row in sudoku_grid:
    #     print(row)
    row_score_sum = 0
    col_score_sum = 0
    box_score_sum = 0

    for i in range(0,len(sudoku_grid)):
        row_score_sum += calc_row_fitness(sudoku_grid,i)

    for i in range(0,len(sudoku_grid)):
        col_score_sum += calc_col_fitness(sudoku_grid,i)

    N = len(sudoku_grid)
    step_size = int(math.sqrt(size))

    for i in range(0, N, step_size):
        for j in range(0, N, step_size):
            box_score_sum += calc_box_fitness(sudoku_grid,i,j,int(math.sqrt(size)))

    # print("row sum: ",row_score_sum)
    # print("column sum: ",col_score_sum)
    # print("boxes sum: ", box_score_sum)
    return row_score_sum+col_score_sum+box_score_sum
def calc_row_fitness(grid,index): # calculates the fitness of a rowat a certain index in the grid
    row_set = set(grid[index])
    return len(row_set)


def calc_col_fitness(grid,index): # calculates the fitness of a column at a certain index in the grid
    # Extract the specified column
    column = [row[index] for row in grid if len(row) > index]
    # Convert the column to a set to find unique elements
    unique_elements = set(column)

    # Return the number of unique elements
    return len(unique_elements)


def calc_box_fitness(matrix, starting_row_index, starting_column_index, box_row_size): # calculates the fitness of each box of size sqrt(n)Xsqrt(n)

    sub_matrix_elements = []
    # Extract the elements of the sub-matrix
    for i in range(starting_row_index, starting_row_index + box_row_size):
        for j in range(starting_column_index, starting_column_index + box_row_size):
            if i < len(matrix) and j < len(matrix[i]):
                sub_matrix_elements.append(matrix[i][j])

    # Convert the list to a set to find unique elements
    unique_elements = set(sub_matrix_elements)

    # Return the number of unique elements
    return len(unique_elements)



def enhance_grid_if_possible(grid,originalGrid,size,currentGridFitness):
    current_enhanced_grid = []
    current_enhanced_grid_fitness = currentGridFitness
    copied_grid = copy.deepcopy(grid)
    for i in range(0,len(grid)-1):
            currentRow = grid[i]
            original_row = originalGrid[i]
            missing_numbers = mut.find_missing_numbers(currentRow)
            duplicate_indexes = mut.find_duplicate_indices(currentRow)
            for idx in duplicate_indexes:
                if (original_row[idx] == 0):
                    for number in missing_numbers:
                        copied_grid[i][idx] = number
                        currentFitness = calc_fitness_sudoku_general(copied_grid)
                        if currentFitness > currentGridFitness:
                            if currentFitness > current_enhanced_grid_fitness:
                                current_enhanced_grid = copy.deepcopy(copied_grid)
                                current_enhanced_grid_fitness = currentFitness

                        copied_grid = copy.deepcopy(grid)


    if(current_enhanced_grid_fitness > currentGridFitness):
        return current_enhanced_grid

    else:
        return grid

# Example Sudoku grid (0 represents empty cells)
input_sudoku_grid_easy1 = [
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0]
]


input_sudoku_grid_easy2 = [
    [1, 0, 0, 4, 8, 9, 0, 0, 6],
    [7, 3, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 1, 2, 5, 9],
    [0, 0, 7, 1, 2, 0, 6, 0, 0],
    [5, 0, 0, 7, 0, 3, 0, 0, 8],
    [0, 0, 6, 0, 9, 5, 7, 0, 0],
    [9, 1, 4, 6, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 3, 7],
    [8, 0, 0, 5, 1, 2, 0, 0, 4]
]

input_sudoku_grid_intermediate1 = [
    [0, 2, 0, 6, 0, 8, 0, 0, 0],
    [5, 8, 0, 0, 0, 9, 7, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [3, 7, 0, 0, 0, 0, 5, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 8, 0, 0, 0, 0, 1, 3],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 9, 8, 0, 0, 0, 3, 6],
    [0, 0, 0, 3, 0, 6, 0, 9, 0]
]

input_sudoku_grid_hard1 =  [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0],
]

input_sudoku_grid_hard2 = [
    [2, 0, 0, 3, 0, 0, 0, 0, 0],
    [8, 0, 4, 0, 6, 2, 0, 0, 3],
    [0, 1, 3, 8, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 2, 0, 3, 9, 0],
    [5, 0, 7, 0, 0, 0, 6, 2, 1],
    [0, 3, 2, 0, 0, 6, 0, 0, 0],
    [0, 2, 0, 0, 0, 9, 1, 4, 0],
    [6, 0, 1, 2, 5, 0, 8, 0, 9],
    [0, 0, 0, 0, 0, 1, 0, 0, 2],
]
small_sudoku_grid = [
    [2, 0, 0, 0],
    [0, 1, 0, 2],
    [0, 0, 3, 0],
    [0, 0, 0, 4]
]

# small_sudoku_grid = [
#     [1, 0, 0, 4],
#     [0, 3, 0, 0],
#     [2, 0, 3, 0],
#     [4, 1, 2, 0]
# ]


def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method,mutation_method,parent_selection_method,problem):

    population = []
    game = input_sudoku_grid_hard2
    optimal_fitness = 243
    for i in range(pop_size):
        if problem == "strings":
            individual = objects.StringIndividual(num_genes)
        if problem == "sudoku":
            # individual = objects.SudokuIndividual(game,len(game))
            # individual.init_random_sudoku_individual_byBlocks(game)
            if i>= 0 and i<220:
                individual = objects.SudokuIndividual(game,len(game))
                individual.init_random_sudoku_individual_byBlocks(game)
            if i>= 220 and i <270:
                # individual = objects.SudokuIndividual(game, len(game))
                # individual.init_random_sudoku_individual_byBlocks(game)
                individual = objects.SudokuIndividual(game, len(game))
                individual.init_random_sudoku_individual_byRows(game)
            if i>= 270 and i <300:
                # individual = objects.SudokuIndividual(game, len(game))
                # individual.init_random_sudoku_individual_byBlocks(game)
                individual = objects.SudokuIndividual(game, len(game))
                individual.init_random_sudoku_individual_byColumns(game)
        if problem == "binpack":
            pass

        population.append(individual)


    generation_avg_fitnesses = []
    generation_avg_SD = []
    cpu_times = []
    elapsed_times = []
    elite_size = int(pop_size * 0.1)

    generation_counter = -1

    for generation in range(max_generations):
        generation_counter += 1
        start_cpu = time.process_time()
        start_elapsed = time.time()

        fitnesses = [fitness_func(individual) for individual in population]

        current_best_indiv_index = 0
        current_best_fitness = 0
        for i in range(0,len(fitnesses)):
            if fitnesses[i] > current_best_fitness:
                current_best_fitness = fitnesses[i]
                current_best_indiv_index = i


        best_indiv = population[current_best_indiv_index]
        best_indiv_grid = best_indiv.grid

        if(current_best_fitness == optimal_fitness):
            print("solution satisfied at generation: ", generation_counter)
            return best_indiv,current_best_fitness


        Statistics_Manager = objects.statistics_manager(fitnesses)
        generation_avg_fitnesses.append(Statistics_Manager.avg_fittness_generation())
        generation_avg_SD.append(Statistics_Manager.Standard_deviation())

        # if  generation%20==0:
        #     Statistics_Manager.norm_and_plot()


        # PARENT SELECTION METHODS
        if problem == "strings":
            elites = parentSelection.elitism(population,pop_size,fitnesses, elite_size)



        if problem == "sudoku":
            # with linear scale
            if parent_selection_method == "rws":
                fitnesses = parentSelection.linear_scaling(5,1,fitnesses)
                elite_indices = ps.roulette_wheel_selection(fitnesses,elite_size)
                elites = [population[i] for i in elite_indices]

            if parent_selection_method == "sus":
                fitnesses = parentSelection.linear_scaling(5,1,fitnesses)
                elite_indices = ps.stochastic_universal_sampling(fitnesses, elite_size)
                elites = [population[i] for i in elite_indices]

            if parent_selection_method == "tournament":

                elite_indices = ps.undeterministic_tournament_selection(fitnesses, elite_size)
                elites = [population[i] for i in elite_indices]

            if parent_selection_method == "ranking":
                index_counter ={}
                for i in range(100):
                    elite_indices = ps.roulette_wheel_selection(fitnesses,elite_size)
                    for i in elite_indices:
                        if i in index_counter:
                            index_counter[i]+=1
                        else:
                            index_counter[i]=1
                sorted_by_values = dict(sorted(index_counter.items(), key=lambda item: item[1]))
                last_n_keys = [key for key, value in sorted_by_values[-elite_size:]]
                elites = [population[i] for i in last_n_keys]







        # increasing age of elites by one since the rest are either children so age zero or dead (Rest of the population)
        for indiv in elites:
            indiv.age+=1


        if problem == "binpack":
            if parent_selection_method == "rws":
                fitnesses = parentSelection.linear_scaling(5,1,fitnesses)
                elite_indices = ps.roulette_wheel_selection(fitnesses,elite_size)
                elites = [population[i] for i in elite_indices]

            if parent_selection_method == "sus":
                fitnesses = parentSelection.linear_scaling(5,1,fitnesses)
                elite_indices = ps.stochastic_universal_sampling(fitnesses,elite_size)
                elites = [population[i] for i in elite_indices]

            if parent_selection_method == "tournament":
                pass


        #CROSSOVER
        offspring = []
        while len(offspring) < pop_size - elite_size:
            if problem == "strings" and crossover_method == "uniform":
               child = com.Uniform(elites,num_genes)

            elif  problem == "strings" and crossover_method == "single":
                child = com.Single(elites,num_genes)

            elif  problem == "strings" and crossover_method == "two":
                child = com.Two(elites,num_genes)

            elif problem == "sudoku" and crossover_method == "pmx":

                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                child_grid =com.pmx_crossover_sudoku_grid_block(parent1,parent2,game) # returns 1 child grid
                #child_grid = com.pmx_crossover_sudoku_grid(parent1, parent2,input_sudoku_grid)  # returns 1 child grid


                child = objects.SudokuIndividual(child_grid,len(game)) # create a new born sudoku individual
                #child.grid = enhance_grid_if_possible(child.grid,game,len(game),calc_fitness_sudoku_general(child.grid))
            elif problem == "sudoku" and crossover_method == "cx":
                pass

            elif problem == "binpack" and crossover_method == "pmx":
                pass
            elif problem == "binpack" and crossover_method == "cx":
                pass

            # MUTATION
            ga_mutation = mutation_rate * sys.maxsize
            if generation_counter > 0:
               ga_mutation =  ga_mutation / (4*generation_counter)

            if random.random() < ga_mutation:
                if problem == "strings":
                    mut.mutate(child)
                if problem == "sudoku" and mutation_method == "inversion" :

                    child.grid = mut.inversion_mutation_sudoku_grid(child.grid,game,len(child.grid))

                    #child.grid = mut.inversion_mutation_sudoku_columns(child.grid,game)



                if problem == "sudoku" and mutation_method == "scramble":
                    # child.grid = mut.scramble_mutation_sudoku_grid(child.grid,input_sudoku_grid)
                    # child_grid_transpoed = com.transpose_matrix(child.grid)
                    # input_sudoku_grid_transposed = com.transpose_matrix(input_sudoku_grid)
                    # result_transposed = mut.scramble_mutation_sudoku_grid(child_grid_transpoed,input_sudoku_grid_transposed)
                    # child.grid = com.transpose_matrix(result_transposed)

                    child.grid = mut.scramble_mutation_sudoku_grid_block(child.grid, game,len(game))


                if problem == "binpack" and mutation_method == "inversion":
                    child.grid = mut.inversion_mutation_sudoku_grid_new(child.grid,game,len(game))
                if problem == "binpack" and mutation_method == "scramble":
                    pass

            #print("current child fitness:", fitness_func(child))
            offspring.append(child)
        population = elites + offspring

        end_cpu = time.process_time()
        end_elapsed = time.time()

        cpu_times.append(end_cpu - start_cpu)
        elapsed_times.append(end_elapsed - start_elapsed)



    objects.plot_distribution(generation_avg_fitnesses, 'Generation', 'AVG', 'Fittness AVG distribution')
    # objects.plot_distribution(generation_avg_SD, 'Generation', 'AVG', 'Standart Deviation')
    # objects.plot_distribution(cpu_times,"Generation","Cpu-time","Ticks")
    # objects.plot_distribution(elapsed_times, "Generation", "Elapsead-time", "Elapsed")


    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)


    return best_individual, best_fitness


bestIndividual,bestFitness = genetic_algorithm(300, 9, calc_fitness_sudoku, 100, 0, "pmx","scramble","tournament","sudoku")

def is_valid_sudoku(grid):
    size = len(grid)
    block_size = int(math.sqrt(size))
    def is_valid_block(block):
        block = [num for num in block if num != 0]
        return len(block) == len(set(block))

    rows_valid = True
    cols_valid = True
    blocks_valid = True

    # Check rows
    for row in grid:
        if not is_valid_block(row):
            rows_valid = False

    # Check columns
    for col in zip(*grid):
        if not is_valid_block(col):
            cols_valid = False

    # Check 3x3 subgrids
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            block = [grid[x][y] for x in range(i, i + block_size) for y in range(j, j + block_size)]
            if not is_valid_block(block):
                blocks_valid = False

    return rows_valid,cols_valid,blocks_valid


if isinstance(bestIndividual,objects.SudokuIndividual):
    print("Best individual: ")
    for row in bestIndividual.grid:
        print(row)

print("Best fitness: ",bestFitness)

rowsvalid,colsvalid,blocksvalid = is_valid_sudoku(bestIndividual.grid)
print("rows valid: ", rowsvalid)
print("cols valid: ",colsvalid)
print("blocks valid: ",blocksvalid)
