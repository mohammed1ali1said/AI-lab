import argparse
import random
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

class statistics_manager:
    def __init__(self, fitnesses):
        self.fitnesses = fitnesses

    def avg_fittness_generation(self):
        sum = 0
        for fitness in self.fitnesses:
            sum += fitness
        return sum / len(self.fitnesses)

    def Standard_deviation(self):
        mean = self.avg_fittness_generation()
        sum = 0
        for fitness in self.fitnesses:
            sum += (fitness - mean) ** 2
        ST = np.sqrt(sum / len(self.fitnesses))
        return ST

    def norm_and_plot(self):
        norm_min = 0
        norm_max = 100
        min_val = 0
        max_val = 13

        div_factor = max_val - min_val
        normalized_fitnesses = []
        for fitness in self.fitnesses:
            normalized = ((fitness - min_val) / (div_factor)) * 100
            normalized_fitnesses.append(normalized)

        fig = plt.figure()
        timer = fig.canvas.new_timer(
            interval=1000)  # creating a timer object and setting an interval of 500 milliseconds

        timer.add_callback(close_event)
        plt.hist(normalized_fitnesses, color='blue', bins=10)
        timer.start()
        plt.show()
        return normalized_fitnesses

def close_event():
    plt.close()


#parents selection
def roulette_wheel_selection(fitnesses,num_parents=2):
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




def linear_scaling(scaling_factor,constant):
    scaled = []
    for fit in fitness:
        scaled.append((fit*scaling_factor+constant))
    return scaled

def Uniform(elites,num_genes):
    parent1 = random.choice(elites)
    parent2 = random.choice(elites)
    child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(num_genes)]
    return child

def Single(elites,num_genes):
    parent1 = random.choice(elites)
    parent2 = random.choice(elites)
    random_index = random.randint(0, num_genes - 1)
    child = parent1[:random_index] + parent2[random_index:]
    return child

def Two(elites,num_genes):
    parent1 = random.choice(elites)
    parent2 = random.choice(elites)
    random_index1 = 0
    random_index2 = 0
    while random_index1 >= random_index2:
          random_index1 = random.randint(0, num_genes - 1)
          random_index2 = random.randint(0, num_genes - 1)

    child = parent1[:random_index1] + parent2[random_index1:random_index2] + parent1[random_index2:]
    return child





def fitness(individual):
    target = list("Hello, world!")
    score = 0
    for i in range(len(individual)):
        if individual[i] == target[i]:
            score += 1
    return score


def mutate(individual):
    target = list("Hello, world!")
    random_position = random.randint(0, len(target) - 1)  # get a random position for character to mutate
    random_char = chr(random.randint(32, 126))
    individual[random_position] = random_char
    return individual


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


def plot_distribution(data, xlabel, ylabel, title):
    # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.plot(data, marker='none', linestyle='-', color='b')

        # Adding labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()



def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method):
    population = []
    for i in range(pop_size):
        individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
        population.append(individual)

    generation_avg_fitnesses = []
    generation_avg_SD = []

    cpu_times = []
    elapsed_times = []

    for generation in range(max_generations):
        start_cpu = time.process_time()
        start_elapsed = time.time()

        fitnesses = [fitness_func(individual) for individual in population]

        Statistics_Manager = statistics_manager(fitnesses)
        generation_avg_fitnesses.append(Statistics_Manager.avg_fittness_generation())
        generation_avg_SD.append(Statistics_Manager.Standard_deviation())
        if  generation%20==0:
            Statistics_Manager.norm_and_plot()

        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        offspring = []
        while len(offspring) < pop_size - elite_size:
            if crossover_method == "uniform":
               child = Uniform(elites,num_genes)

            elif crossover_method == "single":
                child = Single(elites,num_genes)

            elif crossover_method == "two":
                child = Two(elites,num_genes)


            ga_mutation = mutation_rate * sys.maxsize
            if random.random() < ga_mutation:
                mutate(child)

            offspring.append(child)
        population = elites + offspring

        end_cpu = time.process_time()
        end_elapsed = time.time()

        cpu_times.append(end_cpu - start_cpu)
        elapsed_times.append(end_elapsed - start_elapsed)


    plot_distribution(generation_avg_fitnesses, 'Generation', 'AVG', 'Fittness AVG distribution')
    plot_distribution(generation_avg_SD, 'Generation', 'AVG', 'Standart Deviation')
    plot_distribution(cpu_times,"Generation","Cpu-time","Ticks")
    plot_distribution(elapsed_times, "Generation", "Elapsead-time", "Elapsed")

    best_individual = max(population, key=lambda individual: fitness_func(individual))
    best_fitness = fitness_func(best_individual)

    return best_individual, best_fitness


def main():
    parser = argparse.ArgumentParser(description='Genetic Algorithm Parameters')
    parser.add_argument('--pop_size', type=int, default=100, help='Population size')
    parser.add_argument('--max_generations', type=int, default=100, help='Maximum number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.25, help='Mutation rate')
    parser.add_argument('--crossover_method', type=str, default="uniform", choices=["uniform", "single", "two"], help='Crossover method')
    args = parser.parse_args()

    pop_size = args.pop_size
    num_genes = 13
    max_generations = args.max_generations
    mutation_rate = args.mutation_rate
    crossover_method = args.crossover_method

    best_individual, best_fitness = genetic_algorithm(pop_size, num_genes, evaluation, max_generations, mutation_rate, crossover_method)

    print("Best individual:", ''.join(best_individual))
    print("Best fitness:", best_fitness)


if __name__ == "__main__":
    main()

