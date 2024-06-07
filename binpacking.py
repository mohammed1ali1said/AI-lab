import random
import parentSelection  as ps
import crossOverMethods as com

class BinPackingProblem:
    def __init__(self, item_sizes, bin_capacity, num_items):
        self.item_sizes = item_sizes
        self.bin_capacity = bin_capacity
        self.num_items = num_items

# Individual in the population
class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = float('inf')  # Initialize with a high fitness value
        self.age = 0

    def evaluate_fitness(self, fitness_func, problem,opt):
        self.fitness = fitness_func(self.chromosome, problem,opt) + self.age * 10
        return self.fitness

# Genetic algorithm for bin packing
class GeneticAlgorithm:
    def __init__(self, pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method, mutation_method, parent_selection_method, problem, opt,heuristic=None,):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.parent_selection_method = parent_selection_method
        self.problem = problem
        self.heuristic = heuristic
        self.population = []
        self.opt=opt

    def initialize_population(self):
        for _ in range(self.pop_size):
            chromosome = [random.randint(0, 119) for _ in range(self.num_genes)]
            individual = Individual(chromosome)
            individual.evaluate_fitness(self.fitness_func, self.problem,self.opt)
            self.population.append(individual)

    @staticmethod
    def first_fit_heuristic(item_sizes, bin_capacity, max_bins):
        num_items = len(item_sizes)
        chromosome = [-1] * num_items
        bin_sizes = [0] * max_bins

        for i in range(num_items):
            for bin_index in range(max_bins):
                if bin_sizes[bin_index] + item_sizes[i] <= bin_capacity:
                    chromosome[i] = bin_index
                    bin_sizes[bin_index] += item_sizes[i]
                    break

        return chromosome

    @staticmethod
    def worst_fit_heuristic(item_sizes, bin_capacity, max_bins):
        num_items = len(item_sizes)
        chromosome = [-1] * num_items
        bin_sizes = [0] * max_bins

        for i in range(num_items):
            worst_bin = bin_sizes.index(min(bin_sizes))
            if bin_sizes[worst_bin] + item_sizes[i] <= bin_capacity:
                chromosome[i] = worst_bin
                bin_sizes[worst_bin] += item_sizes[i]
            else:
                for bin_index in range(max_bins):
                    if bin_sizes[bin_index] + item_sizes[i] <= bin_capacity:
                        chromosome[i] = bin_index
                        bin_sizes[bin_index] += item_sizes[i]
                        break

        return chromosome

    @staticmethod
    def best_fit_heuristic(item_sizes, bin_capacity, max_bins):
        num_items = len(item_sizes)
        chromosome = [-1] * num_items
        bin_sizes = [0] * max_bins

        for i in range(num_items):
            best_bin = None
            min_space_left = float('inf')
            for bin_index in range(max_bins):
                if bin_sizes[bin_index] + item_sizes[i] <= bin_capacity:
                    space_left = bin_capacity - (bin_sizes[bin_index] + item_sizes[i])
                    if space_left < min_space_left:
                        min_space_left = space_left
                        best_bin = bin_index
            if best_bin is not None:
                chromosome[i] = best_bin
                bin_sizes[best_bin] += item_sizes[i]

        return chromosome

    def select_parents(self):
        return self.parent_selection_method(self.population)

    def crossover(self, parent1, parent2, problem):
        return self.crossover_method(parent1, parent2, problem)

    def mutate(self, individual):
        return self.mutation_method(individual, self.mutation_rate, self.problem.item_sizes, self.problem.bin_capacity)

    def evolve(self):
        self.initialize_population()

        for generation in range(self.max_generations):
            for i in self.population:
                i.age += 1
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2, self.problem)
                self.mutate(child1)
                self.mutate(child2)
                child1.evaluate_fitness(self.fitness_func, self.problem,self.opt)
                child2.evaluate_fitness(self.fitness_func, self.problem,self.opt)
                new_population.extend([child1, child2])
            self.population = sorted(new_population, key=lambda x: x.fitness)[:self.pop_size]
            print(diversity_index(self.population))
            print("")

        best_individual = min(self.population, key=lambda x: x.fitness)

        return best_individual

def fitness_func(chromosome, problem,opt):
    fitness=0
    bin_sizes = [0] * (max(chromosome) + 1)
    for i, bin_index in enumerate(chromosome):
        bin_sizes[bin_index] += problem.item_sizes[i]
        if bin_sizes[bin_index]>150:
            fitness+=100000

    num_bins_used = len([size for size in bin_sizes if size > 0])


    fitness += num_bins_used
    return fitness-opt

def adaptive_fitness_func(chromosome, problem, opt):
    bin_sizes = [0] * (max(chromosome) + 1)
    overflow_penalty = 0
    empty_space_penalty = 0
    num_bins_used = 0

    for i, bin_index in enumerate(chromosome):
        bin_sizes[bin_index] += problem.item_sizes[i]

    for size in bin_sizes:
        if size > 0:
            num_bins_used += 1
            if size > problem.bin_capacity:
                overflow_penalty += (size - problem.bin_capacity)  # Penalize overflow
            else:
                empty_space_penalty += (problem.bin_capacity - size)  # Penalize empty space

    fitness = num_bins_used + overflow_penalty + empty_space_penalty
    return fitness - opt



# crossover methods
# this is a simple Single
def Single(parent1, parent2, problem):
    crossover_point = random.randint(1, len(parent1.chromosome) - 1)
    child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
    child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]

    child1_chromosome = ensure_capacity(child1_chromosome, problem)
    child2_chromosome = ensure_capacity(child2_chromosome, problem)

    child1 = Individual(child1_chromosome)
    child2 = Individual(child2_chromosome)
    return child1, child2


def pmx(parent1, parent2, problem):
    crossover_point1 = random.randint(1, len(parent1.chromosome) - 1)
    crossover_point2 = random.randint(1, len(parent1.chromosome) - 1)
    min1 = min(crossover_point1,crossover_point2)
    max1= max(crossover_point2,crossover_point1)
    child1_chromosome = parent1.chromosome[:min1] + parent2.chromosome[min1:max1] + parent1.chromosome[max1:]
    child2_chromosome = parent2.chromosome[:min1] + parent1.chromosome[min1:max1] + parent2.chromosome[max1:]

    child1_chromosome = ensure_capacity(child1_chromosome, problem)
    child2_chromosome = ensure_capacity(child2_chromosome, problem)

    child1 = Individual(child1_chromosome)
    child2 = Individual(child2_chromosome)
    return child1, child2



def cx(individual1, individual2, problem):
    chromosome_length = len(individual1.chromosome)

    # Select crossover points
    crossover_point1 = random.randint(0, chromosome_length - 1)
    crossover_point2 = random.randint(0, chromosome_length - 1)

    # Ensure crossover points are distinct
    while crossover_point2 == crossover_point1:
        crossover_point2 = random.randint(0, chromosome_length - 1)

    # Sort crossover points
    start_crossover_point, end_crossover_point = sorted([crossover_point1, crossover_point2])

    # Perform crossover
    temp_individual1 = individual1.chromosome[start_crossover_point:end_crossover_point]
    temp_individual2 = individual2.chromosome[start_crossover_point:end_crossover_point]

    individual1.chromosome[start_crossover_point:end_crossover_point] = temp_individual2
    individual2.chromosome[start_crossover_point:end_crossover_point] = temp_individual1

    # Perform bin rearrangement if necessary
    individual1 = rearrange_bins(individual1, bin_capacity)
    individual2 = rearrange_bins(individual2, bin_capacity)

    return individual1, individual2

def rearrange_bins(individual, bin_capacity=150):
    bin_sizes = [0] * (max(individual.chromosome) + 1)
    for bin_index in individual.chromosome:
        bin_sizes[bin_index] += 1

    for i in range(len(bin_sizes)):
        if bin_sizes[i] > bin_capacity:
            overflow = bin_sizes[i] - bin_capacity
            for j in range(len(individual.chromosome) - 1, -1, -1):
                if individual.chromosome[j] == i and overflow > 0:
                    bin_sizes[i] -= 1
                    overflow -= 1
                    individual.chromosome[j] = -1

    # Reallocate items from overflowed bins
    for i in range(len(individual.chromosome)):
        if individual.chromosome[i] == -1:
            for j in range(len(bin_sizes)):
                if bin_sizes[j] + 1 <= bin_capacity:
                    bin_sizes[j] += 1
                    individual.chromosome[i] = j
                    break

    return individual




def mutation_method(individual, mutation_rate, item_sizes, bin_capacity):
    bin_sizes = [0] * (max(individual.chromosome) + 1)
    for i, bin_index in enumerate(individual.chromosome):
        bin_sizes[bin_index] += item_sizes[i]

    for i in range(len(individual.chromosome)):
        if random.random() < mutation_rate:
            new_bin = random.randint(0, len(bin_sizes) - 1)
            if bin_sizes[new_bin] + item_sizes[i] <= bin_capacity:
                old_bin = individual.chromosome[i]
                individual.chromosome[i] = new_bin
                bin_sizes[old_bin] -= item_sizes[i]
                bin_sizes[new_bin] += item_sizes[i]

    # Combine bins if the combined size remains under 150
    for i in range(len(bin_sizes)):
        if bin_sizes[i] < 150:
            for j in range(i + 1, len(bin_sizes)):
                if bin_sizes[j] < 150:
                    combined_size = bin_sizes[i] + bin_sizes[j]
                    if combined_size <= bin_capacity:
                        # Try combining bins temporarily
                        combined_bin_sizes = bin_sizes[:]
                        combined_bin_sizes[i] = combined_size
                        combined_bin_sizes[j] = 0
                        if max(combined_bin_sizes) <= bin_capacity:
                            # Update item positions to the new bin
                            for k in range(len(individual.chromosome)):
                                if individual.chromosome[k] == j:
                                    individual.chromosome[k] = i
                            # Update bin sizes
                            bin_sizes[i] = combined_size
                            bin_sizes[j] = 0
                            break

    return individual



def load_values_from_file(file_path):

    first_three_values = []
    remaining_values = []

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Skip the first line (assuming it's a header like 'u120_00')
        file.readline().strip()

        # Read the second line containing the first three values
        first_three_values = [int(value) for value in file.readline().strip().split()]

        # Read the rest of the lines
        for line in file:
            # Convert the line to an integer and add to the remaining values list
            remaining_values.append(int(line.strip()))

    return first_three_values, remaining_values

# parent selection methods
def tournament(population):
    parent1 = min(random.sample(population, k=36), key=lambda x: x.fitness)
    parent2 = min(random.sample(population, k=36), key=lambda x: x.fitness)
    return parent1, parent2


def rws(population):
    fitness = []
    parentsindex = []
    parents = []

    # Calculate fitness values for individuals in the population
    for individual in population:
        fitness.append(individual.fitness)

    # Apply linear scaling to fitness values
    fitness = ps.linear_scaling(10000, 100, fitness)

    # Remove any empty fitness bins
    fitness = [f for f in fitness if f != 0]

    # Perform roulette wheel selection to choose parents
    parentsindex = ps.roulette_wheel_selection(fitness, 2)

    # Retrieve selected parents based on their indices
    for index in parentsindex:
        parents.append(population[index])

    return parents



def sus(population):
    fitness = []
    parentsindex = []
    parents = []
    for i in population:
        fitness.append(i.fitness)
    fitness = ps.linear_scaling(10,7,fitness)
    parentsindex=ps.stochastic_universal_sampling(fitness,2)
    for i in parentsindex:
        parents.append(population[i])
    return parents



def ensure_capacity(chromosome, problem):
    bin_sizes = [0] * (max(chromosome) + 1)
    for i, bin_index in enumerate(chromosome):
        if bin_index >= len(bin_sizes):
            chromosome[i] = bin_sizes.index(min(bin_sizes))
        bin_sizes[bin_index] += problem.item_sizes[i]

    for i, bin_size in enumerate(bin_sizes):
        if bin_size > problem.bin_capacity:
            for j, bin_index in enumerate(chromosome):
                if bin_index == i:
                    for k in range(len(bin_sizes)):
                        if bin_sizes[k] + problem.item_sizes[j] <= problem.bin_capacity:
                            chromosome[j] = k
                            bin_sizes[i] -= problem.item_sizes[j]
                            bin_sizes[k] += problem.item_sizes[j]
                            break
    return chromosome

ftv,item_sizes = load_values_from_file('try1.txt')

bin_capacity = ftv[0]
num_items = ftv[1]
opt = ftv[2]

problem = BinPackingProblem(item_sizes, bin_capacity, num_items)

def print_total_size_used(chromosome, item_sizes):
    bins = [0] * (max(chromosome) + 1)
    for i, bin_index in enumerate(chromosome):
        bins[bin_index] += item_sizes[i]
    for i, bin_size in enumerate(bins):
        if bin_size!=0:
           print(f"Bin {i}: {bin_size}")
    print(sum(item_sizes),sum(bins))



def number_of_none_empty(chromosome):
    unique = set(chromosome)
    return len(unique)


def diversity_index(population):
    mydict = {}

    for indiv in population:
        size = number_of_none_empty(indiv.chromosome)
        if size in mydict:
            mydict[size]+=1
        else:
            mydict[size]=1
    return mydict




# Using best fit heuristic
ga = GeneticAlgorithm(
    pop_size=150,
    num_genes=num_items,
    fitness_func=fitness_func,
    max_generations=150,
    mutation_rate=0.99,
    crossover_method=cx,
    mutation_method=mutation_method,
    parent_selection_method=tournament,
    problem=problem,
    opt=opt,
    heuristic=GeneticAlgorithm.best_fit_heuristic  # Pass the heuristic here
)

solution_best_fit = ga.evolve()
print(solution_best_fit.chromosome,solution_best_fit.fitness)
print_total_size_used(solution_best_fit.chromosome,item_sizes)