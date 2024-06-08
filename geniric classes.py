import numpy as np
import argparse
import random
import sys
import time
import math
import matplotlib.pyplot as plt

import Objects
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


def top_average_selection_probability_ratio(fitness_values):
    """Calculates the Top-Average Selection Probability Ratio."""
    # Step 1: Calculate total fitness
    total_fitness = sum(fitness_values)

    # Step 2: Determine top fitness
    top_fitness = max(fitness_values)

    # Step 3: Compute average fitness
    average_fitness = total_fitness / len(fitness_values)

    # Step 4: Compute selection probabilities
    selection_prob_top = top_fitness / total_fitness
    selection_prob_avg = average_fitness / total_fitness

    # Step 5: Calculate the ratio
    ratio = selection_prob_top / selection_prob_avg

    return ratio


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


def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method,mutation_method,parent_selection_method,problem,problem_path):
    parameters = {
        'Problem': problem,
        'Population Size': pop_size,
        'Crossover': crossover_method,
        'Mutation Rate': mutation_rate,
        'Mutation Method': mutation_method,
        'Selection Method': parent_selection_method,
        'Max generations': max_generations


    }
    current_bf = 0
    population = []
    game = input_sudoku_grid_easy1
    optimal_fitness = 243
    for i in range(pop_size):
        if problem == "strings":
            individual = objects.StringIndividual(num_genes)
        if problem == "sudoku":
            individual = objects.SudokuIndividual(game,len(game))
            individual.init_random_sudoku_grid(game)


        if problem == "binpack":
            pass

        population.append(individual)


    generation_avg_fitnesses = []
    generation_avg_SD = []
    generation_avg_variance = []
    generation_top_avg_selection_ratio = []
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
                current_bf = fitnesses[i]
                current_best_indiv_index = i




        best_indiv = population[current_best_indiv_index]
        best_indiv_grid = best_indiv.grid
        print("gen: ",generation_counter ,"fitness: ", current_best_fitness)

        if(current_best_fitness == optimal_fitness):
            print("solution satisfied at generation: ", generation_counter)
            for row in best_indiv_grid:
                print(row)
            xLabels = ['Generation','Generation','Generation','Generation','Generation','Generation']
            yLabels = ['AVG','SD','VAR','TR','Cpu-time','Elapsead-time']
            titles = ['Fittness AVG distribution','Standard Deviation','Variance','Top Ratio','Ticks','Elapsed']
            dataSets = [generation_avg_fitnesses, generation_avg_SD, generation_avg_variance,generation_top_avg_selection_ratio, cpu_times, elapsed_times]
            objects.combine_plots(dataSets,xLabels,yLabels,titles,parameters,best_indiv_grid)
            return best_indiv,current_best_fitness


        Statistics_Manager = objects.statistics_manager(fitnesses)
        generation_avg_fitnesses.append(Statistics_Manager.avg_fittness_generation())
        generation_avg_SD.append(Statistics_Manager.Standard_deviation())
        generation_avg_variance.append(Statistics_Manager.Standard_deviation() ** 2)
        generation_top_avg_selection_ratio.append(top_average_selection_probability_ratio(fitnesses))
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

            if parent_selection_method == "elitism":
                elites = parentSelection.elitism(population, pop_size, fitnesses, elite_size)





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

            elif problem == "sudoku" and crossover_method == "cx":
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                child_grid = com.cx_crossover_sudoku(parent1,parent2,game)
                child = objects.SudokuIndividual(child_grid, len(game))  # create a new born sudoku individual
                #pass

            elif problem == "binpack" and crossover_method == "pmx":
                pass
            elif problem == "binpack" and crossover_method == "cx":
                pass

            # MUTATION

            ga_mutation = mutation_rate * sys.maxsize

            if random.random() < ga_mutation:
                if problem == "strings":
                    mut.mutate(child)
                if problem == "sudoku" and mutation_method == "inversion" :

                    random_row_index = random.randint(0,8)
                    original_row = game[random_row_index]
                    current_row = child.grid[random_row_index]
                    mutated_row = mut.inversion_mutation_sudoku_row(current_row,original_row,9)
                    for i in range(0,len(current_row)):
                        child.grid[random_row_index][i] = mutated_row[i]

                if problem == "sudoku" and mutation_method == "scramble":

                            random_number = random.random()
                            if random_number > 0 and random_number < 0.33:
                                child.grid = mut.scramble_mutation_sudoku_grid_block(child.grid, game, len(game))
                            if random_number >= 0.33  and random_number < 0.66:
                                child.grid = mut.scramble_mutation_sudoku_grid(child.grid,game)
                            if random_number >= 0.66 and random_number < 1:
                                child_grid_transpoed = com.transpose_matrix(child.grid)
                                input_sudoku_grid_transposed = com.transpose_matrix(game)
                                result_transposed = mut.scramble_mutation_sudoku_grid(child_grid_transpoed,input_sudoku_grid_transposed)
                                child.grid = com.transpose_matrix(result_transposed)



                if problem == "sudoku" and mutation_method == "replacement":
                    child.grid = mut.replacement_mutation(child.grid,game)
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

    # THESE PLOTS WILL BE PLOTTED IN CASE THE SOLUTION REACHED THE GLOBAL OPTIMA
    # objects.plot_distribution(generation_avg_fitnesses, 'Generation', 'AVG', 'Fittness AVG distribution')
    # objects.plot_distribution(generation_avg_SD, 'Generation', 'SD', 'Standart Deviation')
    # Objects.plot_distribution(generation_avg_variance, 'Generation', 'VAR', 'Variance')
    # objects.plot_distribution(generation_top_avg_selection_ratio, 'Generation', 'avgSelc', 'Top Ratio')
    # objects.plot_distribution(cpu_times, "Generation", "Cpu-time", "Ticks")
    # objects.plot_distribution(elapsed_times, "Generation", "Elapsead-time", "Elapsed")



    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    xLabels = ['Generation', 'Generation', 'Generation', 'Generation', 'Generation', 'Generation']
    yLabels = ['AVG', 'SD', 'VAR', 'TR', 'Cpu-time', 'Elapsead-time']
    titles = ['Fittness AVG distribution', 'Standard Deviation', 'Variance', 'Top Ratio', 'Ticks', 'Elapsed']
    dataSets = [generation_avg_fitnesses, generation_avg_SD, generation_avg_variance,generation_top_avg_selection_ratio, cpu_times, elapsed_times]
    objects.combine_plots(dataSets, xLabels, yLabels, titles, parameters,best_individual.grid)

    return best_individual, best_fitness


# bestIndividual,bestFitness = genetic_algorithm(1500, 9, calc_fitness_sudoku, 300,
#                                                0, "pmx","scramble",
#                                                "elitism","sudoku")



def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Parameters')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--max_generations', type=int, default=100, help='Maximum number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.25, help='Mutation rate')
    parser.add_argument('--crossover_method', type=str, default="uniform", choices=["uniform", "single", "two","pmx","cx"], help='Crossover method')
    parser.add_argument('--mutation_method', type=str, default="scramble",choices=["scramble","inversion"], help='Mutation Method')
    parser.add_argument('--parent_selection', type=str, default="elitism", help='Parent Selection')
    parser.add_argument('--problem', type=str, default="sudoku", help='Problem to test')
    parser.add_argument('--problem_path', type=str, default="//", help='Path to the problem file')
    args = parser.parse_args()

    pop_size = args.pop_size
    num_genes = 13
    max_generations = args.max_generations
    mutation_rate = args.mutation_rate
    crossover_method = args.crossover_method
    mutation_method = args.mutation_method
    problem = args.problem
    parent_selection = args.parent_selection
    problem_path = args.problem_path
    best_individual, best_fitness = genetic_algorithm(2000, 9, calc_fitness_sudoku, 100,
                                                    0, "pmx", "scramble",
                                                    "elitism", "sudoku",problem_path = "//")

    # best_individual, best_fitness = genetic_algorithm(pop_size, num_genes, calc_fitness_sudoku, max_generations,
    #                                                   mutation_rate, crossover_method, mutation_method,
    #                                                   parent_selection, problem)

    # print("Best individual:", ''.join(best_individual))
    # print("Best fitness:", best_fitness)


if __name__ == "__main__":
    main()


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


# if isinstance(bestIndividual,objects.SudokuIndividual):
#     print("Best individual: ")
#     for row in bestIndividual.grid:
#         print(row)
#
# print("Best fitness: ",bestFitness)
#
# rowsvalid,colsvalid,blocksvalid = is_valid_sudoku(bestIndividual.grid)
# print("rows valid: ", rowsvalid)
# print("cols valid: ",colsvalid)
# print("blocks valid: ",blocksvalid)

