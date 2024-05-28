from random import random

import numpy as np


def roulette_wheel_selection(fitnesses,num_parents=100):
    total = sum(fitnesses)
    probs = []
    local_sum = 0
    for fit in fitnesses:
        probs.append((fit / total) + local_sum)
        local_sum += fit / total

    parents = []
    for _ in range(num_parents):
        #generate two random numbers
        rand_num = random.random()
        # use binary to find index
        index = np.searchsorted(probs, rand_num)
        parents.append(index)
    return parents


def stochastic_universal_sampling(fitnesses, num_parents=2):
    total_fitness = sum(fitnesses)
    num_individuals = len(fitnesses)
    spacing = total_fitness / num_parents

    # Choose a random starting point between 0 and spacing
    start = np.random.uniform(0, spacing)
    points = [start + i * spacing for i in range(num_parents)]

    # Compute cumulative fitness array
    cumulative_fitness = np.cumsum(fitnesses)

    selected_parents = []
    for point in points:
        # Use nps binary search to find the index
        index = np.searchsorted(cumulative_fitness, point, side='right')
        selected_parents.append(index % num_individuals)

    return selected_parents


def undeterministic_tournament_selection(fitnesses, num_parents=2, k=2, p=0.75):
    num_individuals = len(fitnesses)
    selected_parents = []

    for _ in range(num_parents):
        # Select k individuals from the population
        tournament_indices = np.random.choice(range(num_individuals), size=k, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # Sort the selected individuals based on their fitness values
        sorted_indices = np.argsort(tournament_fitnesses)[::-1]

        # Determine the probability of selection for each individual based on their rank
        selection_probs = [p * (1 - p) ** i for i in range(k)]

        # Normalize the selection probabilities
        selection_probs /= np.sum(selection_probs)

        # Select an individual based on the calculated probabilities
        selected_index = np.random.choice(sorted_indices, p=selection_probs)
        selected_parents.append(tournament_indices[selected_index])

    return selected_parents

def rescale_by_age(maxgeneration,fitnesses,ages):
    for index in range(len(fitnesses)):
        fitnesses[index] = float(1-(ages[index]/maxgeneration))



def elitism(population,pop_size,fitnesses,elite_size):
    elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
    elites = [population[i] for i in elite_indices]
    return elites


def linear_scaling(scaling_factor,constant,fitnesses):
    scaled = []
    for fit in fitnesses:
        scaled.append((fit*scaling_factor+constant))
    return scaled

