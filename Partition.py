import math
import random

import numpy as np
from matplotlib import pyplot as plt

import Objects as objs
from scipy.cluster.vq import kmeans, vq

def K_means(data, num_clusters=2, visualize=False):
    data = np.array(data, dtype=np.float32)
    data = data.reshape(-1, 1)  # Reshape to one-dimensional data with one feature

    centroids, _ = kmeans(data, num_clusters)

    # Assign each sample to a cluster
    cluster_labels, _ = vq(data, centroids)

    if visualize:
        plt.scatter(data[:, 0], np.zeros_like(data[:, 0]), c=cluster_labels, cmap='viridis')
        plt.scatter(centroids[:, 0], np.zeros_like(centroids[:, 0]), s=200, c='red', marker='x')
        plt.yticks([])  # Remove y-axis ticks
        plt.show()
    return centroids



# CROWDING
def calculate_boltzmann_probability(fitness1, fitness2, T):
    # Calculate the fitness difference
    delta_fitness = fitness1 - fitness2
    # print("fitness1: ", fitness1)
    # print("fitness2: ",fitness2)
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


def calculate_distance_binpack(indiv1_fitness,indiv2_fitness):
    return abs(indiv1_fitness - indiv2_fitness)


def adjust_fitness_with_sharing_binpack(fitnesses, sigma):

    adjusted_fitness = []
    for i in range(len(fitnesses)):
        niche_sum = sum(sharing_function(calculate_distance_binpack(fitnesses[i], fitnesses[j]), sigma) for j in range(len(fitnesses)))
        adjusted_fitness.append(fitnesses[i] / niche_sum if niche_sum > 0 else fitnesses[i])
    return adjusted_fitness


# SPECIATION
# K_means(data=[],num_clusters=10,visualize=True)
