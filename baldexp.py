import random
import statistics
import matplotlib.pyplot as plt
import matplotlib.animation as animation


size = 100
changes =[]

def animate_plot(y_values,ax_tag):
    # Initialize a figure and axis
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-', animated=True)
    ax.set_xlim(0, len(y_values))
    ax.set_ylim(min(y_values) - 1, max(y_values) + 1)

    # Initialize data
    xdata, ydata = [], []

    # Function to initialize the animation
    def init():
        line.set_data([], [])
        return line,

    # Function to update the plot
    def update(frame):
        xdata.append(frame + 1)
        ydata.append(y_values[frame])
        line.set_data(xdata, ydata)
        return line,

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=len(y_values), init_func=init, blit=True, interval=100, repeat=False)
    ax.set_title(ax_tag)

    # Display the plot
    plt.show()

def chromGenerator(size):
    return chromrand(size, ["0", "1", "?"], [0.25, 0.25, 0.50])

def chromrand(size, options, weights):
    chrom = ""
    for i in range(size):
        chrom += random.choices(options, weights=weights)[0]
    return chrom

goal_str = chromrand(100,["0","1"],[0.5,0.5])

# Dummy fitness function to be replaced with actual implementation
def fitness_func(indiv):
    count = 0
    for index,c in enumerate(indiv):
        if c == goal_str[index]:
            count +=1
    return  count

# Individual in the population
class Individual:
    def __init__(self):
        self.chrom = chromGenerator(size=size)
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        self.fitness = fitness_func(self.chrom)
        return self.fitness

    def perform_learning_trials(self, num_trials=100):
        best_chrom = self.chrom
        best_fitness = self.fitness
        starting_point = self.chrom

        for _ in range(num_trials):
            self.learn()  # Perform learning (adaptation) trial

            # Update best chromosome if fitness improves
            if self.fitness > best_fitness:
                best_chrom = self.chrom
                best_fitness = self.fitness
        finishing_point =self.chrom

        change = hamming_distance(starting_point,finishing_point)
        changes.append(change)


        # After all learning trials, update the individual's chromosome and fitness
        self.chrom = best_chrom
        self.fitness = best_fitness

    def learn(self):
        new_chrom = list(self.chrom)  # Convert string to list to modify characters
        for i in range(len(new_chrom)):
            if new_chrom[i] == '?':
            # Randomly choose '0' or '1'
                new_chrom[i] = random.choice(['0', '1'])

        # Update the chromosome with the randomly chosen values
        self.chrom = ''.join(new_chrom)
        self.fitness = fitness_func(self.chrom)



# Example crossover and parent selection methods (unchanged from your code)
def Single(parent1, parent2):
    # Determine crossover point
    crossover_point = random.randint(1, len(parent1.chrom) - 1)

    # Perform crossover
    child1_chromosome = parent1.chrom[:crossover_point] + parent2.chrom[crossover_point:]
    child2_chromosome = parent2.chrom[:crossover_point] + parent1.chrom[crossover_point:]

    # Create new Individual objects for the children
    child1 = Individual()
    child1.chrom = child1_chromosome

    child2 = Individual()
    child2.chrom = child2_chromosome
    child1.evaluate_fitness()
    child2.evaluate_fitness()

    return child1, child2
def tournament(population):
    sample1 = random.sample(list(enumerate(population)), k=36)
    sample2 = random.sample(list(enumerate(population)), k=36)
    parent1_index, parent1 = max(sample1, key=lambda x: x[1].fitness)
    parent2_index, parent2 = max(sample2, key=lambda x: x[1].fitness)
    return parent1, parent2


def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Genetic algorithm
class GeneticAlgorithm:
    def __init__(self, pop_size, num_genes, fitness_func, max_generations, crossover_method, parent_selection_method):
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.fitness_func = fitness_func
        self.max_generations = max_generations
        self.crossover_method = crossover_method
        self.parent_selection_method = parent_selection_method
        self.population = []

    def initialize_population(self):
        for _ in range(self.pop_size):
            self.population.append(Individual())

    def select_parents(self):
        return self.parent_selection_method(self.population)

    def crossover(self, parent1, parent2):
        return self.crossover_method(parent1, parent2)

    def evolve(self):
        self.initialize_population()
        distances = []
        corrects = []
        avg_distances =[]
        avg_corrects = []
        avg_change = []

        for generation in range(self.max_generations):
            # Learning phase (optional)
            distances = []
            corrects = []
            for individual in self.population:
                dist =hamming_distance(individual.chrom,goal_str)
                distances.append(dist)
                corrects.append(100-dist)
                individual.perform_learning_trials()
            avg_distances.append(statistics.mean(distances))
            avg_corrects.append(statistics.mean(corrects))
            avg_change.append(statistics.mean(changes))

            # Evaluate fitness for selection
            for individual in self.population:
                individual.evaluate_fitness()

            # Select parents
            new_population = []
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_parents()

                # Apply crossover to generate offspring
                child1, child2 = self.crossover(parent1, parent2)

                # Add offspring to the new population
                new_population.extend([child1, child2])

            # Replace the old population with the new one
            self.population = new_population

            # Optionally, print the best fitness of the current generation
            best_individual = max(self.population, key=lambda x: x.fitness)
            print(f'Generation {generation}: Best fitness = {best_individual.fitness}')
        animate_plot(avg_corrects,ax_tag="avg_correct_values")
        animate_plot(avg_distances,ax_tag="avg_wrong_values")
        animate_plot(avg_change,ax_tag="changing")


        return best_individual

# Example usage
ga = GeneticAlgorithm(
    pop_size=1000,
    num_genes=size,
    fitness_func=fitness_func,
    max_generations=10,
    crossover_method=Single,
    parent_selection_method=tournament
)

best_individual = ga.evolve()
print("Best individual:", best_individual.chrom, "Fitness:", best_individual.fitness)
print(goal_str)
