import numpy as np
import argparse
import random
import sys
import time
import matplotlib.pyplot as plt
import crossOverMethods as com
import Mutations as mut
import parentSelection
import parentSelection as ps
import  Objects as objects



def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


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








# Example Sudoku grid (0 represents empty cells)
input_sudoku_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]


def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method,mutation_method,parent_selection_method,problem):

    population = []

    for i in range(pop_size):
        if problem == "strings":
            individual = objects.StringIndividual(num_genes)
        if problem == "sudoku":
            individual = objects.SudokuIndividual(input_sudoku_grid)
            individual.init_random_sudoku_individual(input_sudoku_grid)
        if problem == "binpack":
            pass

        population.append(individual)


    generation_avg_fitnesses = []
    generation_avg_SD = []
    cpu_times = []
    elapsed_times = []
    elite_size = int(pop_size * 0.1)

    for generation in range(max_generations):
        start_cpu = time.process_time()
        start_elapsed = time.time()

        fitnesses = [fitness_func(individual) for individual in population]

        Statistics_Manager = objects.statistics_manager(fitnesses)
        generation_avg_fitnesses.append(Statistics_Manager.avg_fittness_generation())
        generation_avg_SD.append(Statistics_Manager.Standard_deviation())

        if  generation%20==0:
            Statistics_Manager.norm_and_plot()


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
                pass
            if parent_selection_method == "sus":
                fitnesses = parentSelection.linear_scaling(5,1,fitnesses)
                elite_indices = ps.stochastic_universal_sampling(fitnesses,elite_size)
                elites = [population[i] for i in elite_indices]
                pass
            if parent_selection_method == "tournament":
                pass
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
                child_grid =com.pmx_crossover_sudoku_grid(parent1,parent2,input_sudoku_grid) # returns 1 child grid
                child = objects.SudokuIndividual(child_grid) # create a new born sudoku individual

            elif problem == "sudoku" and crossover_method == "cx":
                pass

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

                    pass
                if problem == "sudoku" and mutation_method == "scramble":
                    child.grid = mut.scramble_mutation_sudoku_grid(child.grid,input_sudoku_grid)

                if problem == "binpack" and mutation_method == "inversion":
                    pass
                if problem == "binpack" and mutation_method == "scramble":
                    pass

            offspring.append(child)
        population = elites + offspring

        end_cpu = time.process_time()
        end_elapsed = time.time()

        cpu_times.append(end_cpu - start_cpu)
        elapsed_times.append(end_elapsed - start_elapsed)



    object.plot_distribution(generation_avg_fitnesses, 'Generation', 'AVG', 'Fittness AVG distribution')
    object.plot_distribution(generation_avg_SD, 'Generation', 'AVG', 'Standart Deviation')
    object.plot_distribution(cpu_times,"Generation","Cpu-time","Ticks")
    object.plot_distribution(elapsed_times, "Generation", "Elapsead-time", "Elapsed")

    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness
