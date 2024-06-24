import random
import parentSelection  as ps
import crossOverMethods as com
import Objects as objs
import Mutations as mut
import sys
import Partition as partition
import time


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
    def __init__(self, pop_size, num_genes, fitness_func, max_generations, mutation_rate, crossover_method, mutation_method,mutation_control,partition_method, parent_selection_method, problem, opt,heuristic=None,show_results = "true",save_result_counter = 0):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.mutation_control = mutation_control
        self.partition_method = partition_method
        self.parent_selection_method = parent_selection_method
        self.problem = problem
        self.heuristic = heuristic
        self.population = []
        self.opt=opt
        self.show_results = show_results
        self.save_result_counter = save_result_counter
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

    # EVOLVE IN BINPACKING
    def evolve(self):

        parameters = {
            'Problem': "Binpacking",
            'Population Size': self.pop_size,
            'Crossover': (self.crossover_method).__name__,
            'Mutation Rate': self.mutation_rate,
            'Mutation Method': (self.mutation_method).__name__,
            'Mutation Control': self.mutation_control,
            'Partition Method': self.partition_method,
            'Selection Method': (self.parent_selection_method).__name__,
            'Max generations': self.max_generations

        }

        self.initialize_population()

        generation_counter = -1
        first_gen_avg_fitness = 0
        avg_gen_fitness = 0
        # PARAMETERS FOR HYPER MUTAION
        hyper_mutation_state = "waiting"  # this variable is used in the hyper mutation section if it was chosen
        thm_best_fit_tracker = []  # this list is made to track the best fitnesses of the last x generations.
        thm_generations = 20  # track the last 20 generations
        thm_generation_counter = 0  # start from 0 and end at 20 if the hyper mutation needed.
        thm_thresh = 0.5

        original_mutation_rate = 0

        generation_avg_fitnesses = []
        generation_avg_SD = []
        generation_avg_variance = []
        generation_top_avg_selection_ratio = []
        cpu_times = []
        elapsed_times = []

        for generation in range(self.max_generations):
            start_cpu = time.process_time()
            start_elapsed = time.time()
            population_fitnesses = []
            for indiv in self.population:
                population_fitnesses.append(indiv.fitness)

            if(generation%10==0):
                partition.K_means(population_fitnesses,2,True)

            generation_counter += 1
            print("Gen: ", generation_counter)
            if self.partition_method == ("sharing") :  # adjust the fitnesses from the beggining based on fitness sharing method
                SIGMA = 5
                # print("original fitnesses :", fitnesses)
                adjusted_fitnesses = partition.adjust_fitness_with_sharing_binpack(population_fitnesses, SIGMA)
                fitnesses = adjusted_fitnesses.copy()

                for indiv, adjusted_fitness in zip(self.population, adjusted_fitnesses):
                    indiv.fitness = adjusted_fitness

            Statistics_Manager = objs.statistics_manager(population_fitnesses)
            generation_avg_fitnesses.append(Statistics_Manager.avg_fittness_generation())
            generation_avg_SD.append(Statistics_Manager.Standard_deviation())
            generation_avg_variance.append(Statistics_Manager.Standard_deviation() ** 2)
            generation_top_avg_selection_ratio.append(top_average_selection_probability_ratio(population_fitnesses))


            avg_gen_fitness = 0
            fitness_sum = 0
            current_best_fitness = 10000
            for indiv in self.population:
                fitness_sum += indiv.fitness
                if indiv.fitness < current_best_fitness:
                    current_best_fitness = indiv.fitness
            avg_gen_fitness = fitness_sum/len(self.population)
            if generation_counter == 0:
                first_gen_avg_fitness = avg_gen_fitness
                original_mutation_rate = self.mutation_rate
            for i in self.population:
                i.age += 1
            new_population = []
            current_generation = generation_counter
            # PARENT SELECTION BINPACK
            parent1_index = 0
            parent2_index = 0
            for _ in range(self.pop_size // 2):

                #print("Parent selection method:", self.parent_selection_method.__name__)
                if self.parent_selection_method.__name__ == "tournament" :

                    paren1_index,parent2_index = self.select_parents()
                    parent1 = self.population[parent1_index]
                    parent2 = self.population[parent2_index]
                elif self.parent_selection_method.__name__ != "tournament":

                    parent1, parent2 = self.select_parents()

                child1, child2 = self.crossover(parent1, parent2, self.problem)
                # MUTATION CONTROL
                if (self.mutation_control == "non_uniform"):  # decreases the mutation rate linearly with generations

                    if generation_counter >= 1:
                        self.mutation_rate = self.mutation_rate / (generation_counter)

                if self.mutation_control == "adaptive":  # decreases the mutation rate as the avg fitness decreases (gets better)

                    self.mutation_rate = avg_gen_fitness / first_gen_avg_fitness

                if self.mutation_control == "THM":

                    if (hyper_mutation_state == "waiting") and (len(thm_best_fit_tracker) < thm_generations):  # the first 20 generations
                        if current_generation == generation_counter:
                            print("waiting , currently in the first 20 gens")
                            current_generation = -1
                            thm_best_fit_tracker.append(current_best_fitness)

                    if (hyper_mutation_state == "waiting") and (
                            len(thm_best_fit_tracker) == thm_generations):  # after the list is full, keep tracking the last 20 generations
                        hyper_mutation_needed = "false"
                        if current_generation == generation_counter:
                            hyper_mutation_needed = mut.apply_thm_test_binpack(thm_best_fit_tracker, thm_thresh)
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
                                print(
                                    "hyper mutation period has ended,going back to waiting state, and mutation rate going back to normal")
                                current_generation = -1
                                hyper_mutation_state = "waiting"
                                thm_generation_counter = 0
                                self.mutation_rate = original_mutation_rate

                        if current_generation == generation_counter:
                            print("hyper mutation in progress")
                            current_generation = -1
                            thm_best_fit_tracker.pop(0)
                            thm_best_fit_tracker.append(current_best_fitness)
                            thm_generation_counter += 1


                if self.mutation_control != "self_adaptive":
                    self.mutate(child1)
                    self.mutate(child2)
                child1.evaluate_fitness(self.fitness_func, self.problem,self.opt)
                child2.evaluate_fitness(self.fitness_func, self.problem,self.opt)

                if self.partition_method == "crowding":

                    # 1. calculate the probability of replacing child with parent
                    child1_fitness = child1.fitness
                    parent1_fitness = parent1.fitness
                    boltzman_prob = partition.calculate_boltzmann_probability(parent1_fitness, child1_fitness, 5)
                    random_number = random.random()

                    if random_number <= boltzman_prob:  # replacing will happen
                        self.population[parent1_index] = child1

                new_population.extend([child1, child2])
            self.population = sorted(new_population, key=lambda x: x.fitness)[:self.pop_size]
            # print(diversity_index(self.population))
            # print("")

            # IN THIS MUTATION CONTROL METHOD, WE DISABLE THE NORMAL MUTATION FOR CHILDREN, AND WE DO THE MUTATION INDIVIDUALLY INSIDE THIS BLOCk
            if self.mutation_control == "self_adaptive":
                print("chose self adaptive mutation control, treating each individual now..")

                alpha = 0.5  # multiply mutation rate by 0.5 for strong individuals
                beta = 2  # mutation rate by 2 for weak individuals
                # 1. iterate over the population
                for indiv in self.population:
                    # 2. for each individual calculate the relative fitness
                    indiv.relative_fitness = (indiv.fitness) / (avg_gen_fitness)

                    if indiv.relative_fitness <= 1:  # weak individual
                        mutation_rate = self.mutation_rate * beta
                        ga_mutation = mutation_rate * sys.maxsize

                    if indiv.relative_fitness > 1:  # strong individual
                        mutation_rate = self.mutation_rate * alpha
                        ga_mutation = mutation_rate * sys.maxsize
                        if (random.random() < ga_mutation):
                            if  self.mutation_method == "scramble":
                                self.mutate(child1)
                                child1.evaluate_fitness(self.fitness_func, self.problem, self.opt)

            end_cpu = time.process_time()
            end_elapsed = time.time()


            cpu_times.append(end_cpu - start_cpu)
            elapsed_times.append(end_elapsed - start_elapsed)
        if self.show_results == "true":
            xLabels = ['Generation', 'Generation', 'Generation', 'Generation', 'Generation', 'Generation']
            yLabels = ['AVG', 'SD', 'VAR', 'TR', 'Cpu-time', 'Elapsead-time']
            titles = ['Fittness AVG distribution', 'Standard Deviation', 'Variance', 'Top Ratio', 'Ticks', 'Elapsed']
            dataSets = [generation_avg_fitnesses, generation_avg_SD, generation_avg_variance,
                        generation_top_avg_selection_ratio, cpu_times, elapsed_times]
            objs.combine_plots(dataSets, xLabels, yLabels, titles, parameters, 'none',
                                  r'C:\Users\Administrator\Desktop\Ai-lab2\Report\binpack_result', self.save_result_counter)
        best_individual = min(self.population, key=lambda x: x.fitness)

        return best_individual

def fitness_func(chromosome, problem,opt):
    fitness=0
    bin_sizes = [0] * (max(chromosome) + 1)
    for i, bin_index in enumerate(chromosome):
        bin_sizes[bin_index] += problem.item_sizes[i]
        if bin_sizes[bin_index]>150:
            fitness+=100

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

def Two(parent1,parent2,problem):
    crossover_point1 = random.randint(1, len(parent1.chromosome) - 2)
    crossover_point2 = random.randint(1, len(parent1.chromosome) - 2)
    min1 = min(crossover_point1,crossover_point2)
    max1 = max(crossover_point2,crossover_point1)

    child1_chromosome = parent1.chromosome[:min1] + parent2.chromosome[min1:max1]+ parent1.chromosome[max1:]
    child2_chromosome = parent2.chromosome[:min1] + parent1.chromosome[min1:max1]+ parent2.chromosome[max1:]

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

#parent selection methods
# def tournament(population):
#
#     parent1 = min(random.sample(population, k=36), key=lambda x: x.fitness)
#     parent2 = min(random.sample(population, k=36), key=lambda x: x.fitness)
#     return parent1, parent2


def tournament(population):

    # Sample 36 individuals from the population
    sample1 = random.sample(list(enumerate(population)), k=36)
    sample2 = random.sample(list(enumerate(population)), k=36)

    # Find the individual with the minimum fitness in each sample
    parent1_index, parent1 = min(sample1, key=lambda x: x[1].fitness)
    parent2_index, parent2 = min(sample2, key=lambda x: x[1].fitness)

    # Return the parents and their indexes
    return parent1_index,parent2_index




# def tournament_new(population):
#     print("choosing tournament_new")
#     # Sample 36 individuals from the population
#     sample1 = random.sample(list(enumerate(population)), k=36)
#     sample2 = random.sample(list(enumerate(population)), k=36)
#
#     # Find the individual with the minimum fitness in each sample
#     parent1_index, parent1 = min(sample1, key=lambda x: x[1].fitness)
#     parent2_index, parent2 = min(sample2, key=lambda x: x[1].fitness)
#
#     # Return the parents and their indexes
#     return (parent1_index, parent1), (parent2_index, parent2)
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

ftv,item_sizes = load_values_from_file('try1')

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
            pass
           # print(f"Bin {i}: {bin_size}")
    # print(sum(item_sizes),sum(bins))



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
    max_generations=50,
    mutation_rate=0.5,
    crossover_method=Two,
    mutation_method=mutation_method,
    mutation_control='non_uniform',
    parent_selection_method=tournament,
    partition_method='crowding',
    problem=problem,
    opt=opt,
    heuristic=GeneticAlgorithm.best_fit_heuristic  # Pass the heuristic here
)

solution_best_fit = ga.evolve()
print(solution_best_fit.chromosome,solution_best_fit.fitness)
print_total_size_used(solution_best_fit.chromosome,item_sizes)
