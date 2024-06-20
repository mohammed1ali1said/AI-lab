import importlib

import numpy as np
import argparse
import random
import sys
import time
import math
import matplotlib.pyplot as plt
import binpacking as bp
import Partition as partition
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
input_sudoku_grid_impossible = [
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 0, 0, 3],
    [0, 7, 4, 0, 8, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 2],
    [0, 8, 0, 0, 4, 0, 0, 1, 0],
    [6, 0, 0, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 7, 8, 0],
    [5, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 4, 0],
]
small_sudoku_grid = [
    [2, 0, 0, 0],
    [0, 1, 0, 2],
    [0, 0, 3, 0],
    [0, 0, 0, 4]
]


# GENETIC ALGORITHM DEF
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method,mutation_method,mutation_control,partition_method,parent_selection_method
                      ,problem,problem_path,grid,show_results = "false",save_result_counter = 0):
    parameters = {
        'Problem' : problem,
        'Population Size': pop_size,
        'Crossover': crossover_method,
        'Mutation Rate': mutation_rate,
        'Mutation Method': mutation_method,
        'Mutation Control': mutation_control,
        'Partition Method': partition_method,
        'Selection Method': parent_selection_method,
        'Max generations': max_generations

    }
    original_mutation_rate = mutation_rate
    current_bf = 0
    population = []
    game = input_sudoku_grid_easy1
    if problem == "sudoku":
        optimal_fitness = 243
    if problem == "binpack":
        pass

    for i in range(pop_size):
        if problem == "strings":
            fitness_func = evaluation
            individual = objects.StringIndividual(num_genes)

        if problem == "sudoku":
            if grid=="hard2":
                game = input_sudoku_grid_hard2
            elif grid =="intermediate1":
                game = input_sudoku_grid_intermediate1
            elif grid=="hard1":
                game = input_sudoku_grid_hard1
            elif grid=="easy2":
                game = input_sudoku_grid_easy2
            elif grid=="impossible":     # thanos
                game = input_sudoku_grid_impossible


            fitness_func = calc_fitness_sudoku
            born_fitness = calc_fitness_sudoku_general(game)
            individual = objects.SudokuIndividual(game,len(game),born_fitness)
            individual.init_random_sudoku_grid(game)

        # BINPACK PROBLEM
        if problem == "binpack":
            path = problem_path
            try:
                ftv,item_sizes = bp.load_values_from_file(path)

            except:
                print("no path")
                pass


            bin_capacity = ftv[0]
            num_items = ftv[1]
            opt = ftv[2]
            problem1 = bp.BinPackingProblem(item_sizes, bin_capacity, num_items)
            if fitness_func=="adaptive":
                fitness_func = bp.adaptive_fitness_func
            else:
                fitness_func = bp.fitness_func

            if parent_selection_method == "rws":
                parent_selection_method = bp.rws
            elif parent_selection_method == "sus":
                 parent_selection_method =bp.sus
            else:
                parent_selection_method=bp.tournament
            if crossover_method =="Single":
               crossover_method = bp.Single
            elif crossover_method =="Two":
                crossover_method = bp.Two
            elif crossover_method =="pmx":
                crossover_method =bp.pmx
            elif crossover_method == "cx":
                crossover_method = bp.cx

            ga = bp.GeneticAlgorithm(
            pop_size=pop_size,
            num_genes=num_items,
            fitness_func=fitness_func,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            crossover_method=crossover_method,
            mutation_method=bp.mutation_method,
            mutation_control=mutation_control,
            partition_method =partition_method,
            parent_selection_method=parent_selection_method,
            problem=problem1,
            opt=opt,
            heuristic= bp.GeneticAlgorithm.best_fit_heuristic,  # Pass the heuristic here
            save_result_counter= save_result_counter
            )
            sol=ga.evolve()
            print(sol.chromosome,sol.fitness)
            return 1

        population.append(individual)


    generation_avg_fitnesses = []
    generation_avg_SD = []
    generation_avg_variance = []
    generation_top_avg_selection_ratio = []
    cpu_times = []
    elapsed_times = []
    elite_size = int(pop_size * 0.1)

    generation_counter = -1

    # PARAMETERS FOR HYPER MUTAION
    hyper_mutation_state = "waiting"  # this variable is used in the hyper mutation section if it was chosen
    thm_best_fit_tracker = []  # this list is made to track the best fitnesses of the last x generations.
    thm_generations = 20  # track the last 20 generations
    thm_generation_counter = 0  # start from 0 and end at 20 if the hyper mutation needed.
    thm_thresh = 0.5

    # GENERATION LOOP STARTING
    for generation in range(max_generations):
        generation_counter += 1
        start_cpu = time.process_time()
        start_elapsed = time.time()

        fitnesses = [fitness_func(individual) for individual in population]
        if partition_method == ("sharing") and problem == "sudoku": # adjust the fitnesses from the beggining based on fitness sharing method
            SIGMA = 5
            #print("original fitnesses :", fitnesses)
            population_grids = []
            for indiv in population:
                population_grids.append(indiv.grid)

            adjusted_fitnesses = partition.adjust_fitness_with_sharing(population_grids,fitnesses,SIGMA)
            fitnesses = adjusted_fitnesses.copy()
            for indiv,adjusted_fitness in zip(population,adjusted_fitnesses):
                indiv.fitness = adjusted_fitness
            #print("adjusted fitnesses :", fitnesses)







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
        # for row in best_indiv_grid:
        #     print(row)
        if(current_best_fitness == optimal_fitness):
            print("solution satisfied at generation: ", generation_counter)
            for row in best_indiv_grid:
                print(row)
            if show_results == "true":
                xLabels = ['Generation','Generation','Generation','Generation','Generation','Generation']
                yLabels = ['AVG','SD','VAR','TR','Cpu-time','Elapsead-time']
                titles = ['Fittness AVG distribution','Standard Deviation','Variance','Top Ratio','Ticks','Elapsed']
                dataSets = [generation_avg_fitnesses, generation_avg_SD, generation_avg_variance,generation_top_avg_selection_ratio, cpu_times, elapsed_times]
                objects.combine_plots(dataSets, xLabels, yLabels, titles, parameters,'save',r'C:\Users\Administrator\Desktop\Ai-lab2\Report\sudoku_result',save_result_counter)
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



        current_generation = generation_counter
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

                parent1_index,parent1 = random.choice(list(enumerate(elites)))
                parent2_index,parent2 = random.choice(list(enumerate(elites)))

                child_grid =com.pmx_crossover_sudoku_grid_block(parent1,parent2,game) # returns 1 child grid
                #child_grid = com.pmx_crossover_sudoku_grid(parent1, parent2,input_sudoku_grid)  # returns 1 child grid

                born_fitness = calc_fitness_sudoku_general(child_grid)
                child = objects.SudokuIndividual(child_grid,len(game),born_fitness) # create a new born sudoku individual

            elif problem == "sudoku" and crossover_method == "cx":
                # parent1_index, parent1 = random.choice(list(enumerate(elites)))
                # parent2_index, parent2 = random.choice(list(enumerate(elites)))
                # child_grid = com.cx_crossover_sudoku(parent1,parent2,game)
                # child = objects.SudokuIndividual(child_grid, len(game))  # create a new born sudoku individual
                pass

            # MUTATION CONTROL
            if(mutation_control == "non_uniform"): # decreases the mutation rate linearly with generations

                if generation_counter >= 1:
                    mutation_rate = mutation_rate / (generation_counter)

            if mutation_control == "adaptive": # decreases the mutation rate as the avg fitness increases
                gen_avg_fitness = Statistics_Manager.avg_fittness_generation()
                mutation_rate = (optimal_fitness - gen_avg_fitness) / optimal_fitness



            if mutation_control == "THM":

                if (hyper_mutation_state == "waiting") and (len(thm_best_fit_tracker) < thm_generations): # the first 20 generations
                    if current_generation == generation_counter:
                        print("waiting , currently in the first 20 gens")
                        current_generation = -1
                        thm_best_fit_tracker.append(current_best_fitness)


                if (hyper_mutation_state == "waiting") and (len(thm_best_fit_tracker) == thm_generations): # after the list is full, keep tracking the last 20 generations
                    hyper_mutation_needed = "false"
                    if current_generation == generation_counter:
                        hyper_mutation_needed = mut.apply_thm_test(thm_best_fit_tracker,thm_thresh)
                    if hyper_mutation_needed == "true":
                        if current_generation == generation_counter:
                            print("now in the in_progress phase, changing mutation rate  !")
                            current_generation = -1
                            mutation_rate = 0.9

                            hyper_mutation_state = "in_progress"
                    else:
                        if current_generation == generation_counter:
                            print("waiting , more than 20 generations passed")
                            current_generation = -1
                            thm_best_fit_tracker.pop(0)
                            thm_best_fit_tracker.append(current_best_fitness)


                if hyper_mutation_state == "in_progress":
                    if thm_generation_counter == thm_generations:
                        if current_generation == generation_counter:
                            print("hyper mutation period has ended,going back to waiting state, and mutation rate going back to normal")
                            current_generation = -1
                            hyper_mutation_state = "waiting"
                            thm_generation_counter = 0
                            mutation_rate = original_mutation_rate

                    if current_generation == generation_counter:
                        print("hyper mutation in progress")
                        current_generation = -1
                        thm_best_fit_tracker.pop(0)
                        thm_best_fit_tracker.append(current_best_fitness)
                        thm_generation_counter += 1



            ga_mutation = mutation_rate * sys.maxsize

            # MUTATION
            if (random.random() < ga_mutation) and mutation_control != "self_adaptive":


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

                            child.grid = mut.scramble_mutation_sudoku_grid_block(child.grid, game, len(game))
                            indiv_ctr = 0
                            # random_number = random.random()
                            # if random_number > 0 and random_number < 0.33:
                            #     child.grid = mut.scramble_mutation_sudoku_grid_block(child.grid, game, len(game))
                            # if random_number >= 0.33  and random_number < 0.66:
                            #     child.grid = mut.scramble_mutation_sudoku_grid(child.grid,game)
                            # if random_number >= 0.66 and random_number < 1:
                            #     child_grid_transpoed = com.transpose_matrix(child.grid)
                            #     input_sudoku_grid_transposed = com.transpose_matrix(game)
                            #     result_transposed = mut.scramble_mutation_sudoku_grid(child_grid_transpoed,input_sudoku_grid_transposed)
                            #     child.grid = com.transpose_matrix(result_transposed)



                if problem == "sudoku" and mutation_method == "replacement":
                    child.grid = mut.replacement_mutation(child.grid,game)


            # PARTITION
            if partition_method == "crowding":

                # 1. calculate the probability of replacing child with parent
                child_fitness = child.fitness
                parent1_fitness = parent1.fitness
                boltzman_prob = partition.calculate_boltzmann_probability(parent1_fitness,child_fitness,5)
                random_number = random.random()

                if random_number <= boltzman_prob: # replacing will happen
                    elites[parent1_index] = child

            offspring.append(child)
        population = elites + offspring

        # IN THIS MUTATION CONTROL METHOD, WE DISABLE THE NORMAL MUTATION FOR CHILDREN, AND WE DO THE MUTATION INDIVIDUALLY INSIDE THIS BLOCk
        if mutation_control == "self_adaptive":

            print("chose self adaptive mutation control, treating each individual now..")
            gen_avg_fitness = Statistics_Manager.avg_fittness_generation()
            alpha = 0.5  # multiply mutation rate by 0.5 for strong individuals
            beta = 2  # mutation rate by 2 for weak individuals
            # 1. iterate over the population
            for indiv in population:
                # 2. for each individual calculate the relative fitness
                indiv.relative_fitness = (indiv.fitness) / (gen_avg_fitness)

                if indiv.relative_fitness <= 1:  # weak individual
                    mutation_rate = mutation_rate * beta
                    ga_mutation = mutation_rate * sys.maxsize

                if indiv.relative_fitness > 1:  # strong individual
                    mutation_rate = mutation_rate * alpha
                    ga_mutation = mutation_rate * sys.maxsize
                    if (random.random() < ga_mutation):
                        if problem == "sudoku" and mutation_method == "scramble":
                            child.grid = mut.scramble_mutation_sudoku_grid_block(child.grid, game, len(game))


        end_cpu = time.process_time()
        end_elapsed = time.time()

        cpu_times.append(end_cpu - start_cpu)
        elapsed_times.append(end_elapsed - start_elapsed)



    if show_results == "true":

        xLabels = ['Generation', 'Generation', 'Generation', 'Generation', 'Generation', 'Generation']
        yLabels = ['AVG', 'SD', 'VAR', 'TR', 'Cpu-time', 'Elapsead-time']
        titles = ['Fittness AVG distribution', 'Standard Deviation', 'Variance', 'Top Ratio', 'Ticks', 'Elapsed']
        dataSets = [generation_avg_fitnesses, generation_avg_SD, generation_avg_variance,generation_top_avg_selection_ratio, cpu_times, elapsed_times]
        objects.combine_plots(dataSets, xLabels, yLabels, titles, parameters,'save',r'C:\Users\Administrator\Desktop\Ai-lab2\Report\sudoku_result',save_result_counter)

    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    print("best fitness: ", best_fitness)
    for row in best_individual.grid:
        print(row)
    return best_individual, best_fitness




def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Parameters')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--max_generations', type=int, default=100, help='Maximum number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0, help='Mutation rate')
    parser.add_argument('--crossover_method', type=str, default="pmx", choices=["uniform", "single", "two","pmx","cx"], help='Crossover method')
    parser.add_argument('--mutation_method', type=str, default="scramble",choices=["scramble","inversion"], help='Mutation Method')
    parser.add_argument('--mutation_control', type=str, default="basic",
                        choices=["basic", "non_uniform","adaptive","THM","self_adaptive"],help='Mutation Control')
    parser.add_argument('--parent_selection', type=str, default="elitism", help='Parent Selection')
    parser.add_argument('--partition_method',  type=str, default="crowding",choices=["none","crowding","sharing","speciation"], help='Partition Method')
    parser.add_argument('--problem', type=str, default="sudoku", help='Problem to test')
    parser.add_argument('--problem_path', type=str, default="try1.txt", help='Path to the problem file')
    parser.add_argument('--sudoku_grid',type=str,default='easy1',help='sudoku grid')
    parser.add_argument('--fitness_func',type=str,default="static")
    args = parser.parse_args()

    pop_size = args.pop_size
    num_genes = 13
    max_generations = args.max_generations
    mutation_rate = args.mutation_rate
    crossover_method = args.crossover_method
    mutation_method = args.mutation_method
    mutation_control = args.mutation_control
    partition_method = args.partition_method
    problem = args.problem
    parent_selection = args.parent_selection
    problem_path = args.problem_path
    grid = args.sudoku_grid
    fitness_func= args.fitness_func



    # GENETIC ALGORITHM CALL
    genetic_algorithm(pop_size=1500, num_genes=num_genes,max_generations= 100,
                      mutation_rate=0.15,crossover_method= "pmx",mutation_method= "scramble",
                      mutation_control = "basic",partition_method = "none",parent_selection_method="elitism",problem_path= problem_path,problem="sudoku",
                      fitness_func=fitness_func,grid="easy1",show_results = "true",save_result_counter=0)
    return -1



if __name__ == "__main__":
    main()



