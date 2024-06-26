import copy
import math
import random

import numpy as np
from matplotlib import pyplot as plt


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

class StringIndividual:
      def __init__(self,num_genes):
          self.str = [chr(random.randint(32, 126)) for j in range(num_genes)]
          self.age = 0


class SudokuIndividual:

    def __init__(self,grid,size,born_fitness):
        self.size = size
        self.grid = grid
        self.age = 0
        self.fitness = born_fitness
        self.relative_fitness = 1
        pass

    def init_random_sudoku_individual_byRows(self, input_grid):

        # Create a deep copy of the input grid to modify
        new_grid = [row[:] for row in input_grid]

        for row in new_grid:
            # Determine which numbers are already in the row
            existing_numbers = set(row) - {0}
            # Create a list of possible numbers for the row
            possible_numbers = [num for num in range(1, self.size+1) if num not in existing_numbers]

            for i in range(len(row)):
                if row[i] == 0:
                    # Choose a random number from the possible numbers
                    number = random.choice(possible_numbers)
                    row[i] = number
                    # Remove the chosen number from the possible numbers
                    possible_numbers.remove(number)

        self.grid = new_grid

    def init_random_sudoku_individual_byColumns(self, input_grid):
        # Create a deep copy of the input grid to modify
        new_grid = [row[:] for row in input_grid]

        # Transpose the grid to make columns accessible as rows
        transposed_grid = list(map(list, zip(*new_grid)))

        for col in transposed_grid:
            # Determine which numbers are already in the column
            existing_numbers = set(col) - {0}
            # Create a list of possible numbers for the column
            possible_numbers = [num for num in range(1, self.size+1) if num not in existing_numbers]

            for i in range(len(col)):
                if col[i] == 0:
                    # Choose a random number from the possible numbers
                    number = random.choice(possible_numbers)
                    col[i] = number
                    # Remove the chosen number from the possible numbers
                    possible_numbers.remove(number)

        # Transpose the grid back to its original form
        new_grid = list(map(list, zip(*transposed_grid)))

        self.grid = new_grid

    def init_random_sudoku_individual_byBlocks(self, input_grid):
        # Create a deep copy of the input grid to modify
        new_grid = [row[:] for row in input_grid]
        block_size = int(self.size ** 0.5)

        for block_row in range(0, self.size, block_size):
            for block_col in range(0, self.size, block_size):
                # Determine which numbers are already in the block
                existing_numbers = set()
                for i in range(block_size):
                    for j in range(block_size):
                        if new_grid[block_row + i][block_col + j] != 0:
                            existing_numbers.add(new_grid[block_row + i][block_col + j])

                # Create a list of possible numbers for the block
                possible_numbers = [num for num in range(1, self.size + 1) if num not in existing_numbers]

                for i in range(block_size):
                    for j in range(block_size):
                        if new_grid[block_row + i][block_col + j] == 0:
                            # Choose a random number from the possible numbers
                            number = random.choice(possible_numbers)
                            new_grid[block_row + i][block_col + j] = number
                            # Remove the chosen number from the possible numbers
                            possible_numbers.remove(number)

        self.grid = new_grid

    def init_random_sudoku_grid(self,input_grid):
        # Create a deep copy of the input grid to modify
        new_grid = [row[:] for row in input_grid]
        for i in range(len(new_grid)):
            for j in range(len(new_grid[i])):
                if new_grid[i][j] == 0:  # Assuming empty cells are represented by 0
                    new_grid[i][j] = random.randint(1, 9)
        self.grid =   new_grid
    def print_sudoku_grid(self,input_grid):
        for row in input_grid:
            print(row)
def close_event():
    plt.close()




def plot_distribution(ax, data, xlabel, ylabel, title):
    """Creates a plot on the given axis."""
    ax.plot(data, marker='none', linestyle='-', color='b')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_parameters(ax, parameters):
    """Creates a text plot for the given parameters on the given axis."""
    param_text = "\n".join([f"{key}: {value}" for key, value in parameters.items()])
    ax.text(0.5, 0.5, param_text, fontsize=12, ha='center', va='center', wrap=True)
    ax.axis('off')
    ax.set_title('Algorithm Parameters')

# def combine_plots(datasets, xlabels, ylabels, titles, parameters):
#     """Combines multiple plots into a single image, including parameters."""
#     # Number of plots (plus one for parameters)
#     n = len(datasets) + 1
#
#     # Determine the grid size
#     cols = 3
#     rows = math.ceil(n / cols)
#
#     # Create a figure to hold the subplots
#     fig, axs = plt.subplots(rows, cols, figsize=(18, rows * 4))
#
#     # Flatten the array of axes for easy iteration
#     axs = axs.flatten()
#
#     # Plot the parameters in the first subplot
#     plot_parameters(axs[0], parameters)
#
#     # Create each subplot for the datasets
#     for i in range(1, n):
#         plot_distribution(axs[i], datasets[i-1], xlabels[i-1], ylabels[i-1], titles[i-1])
#
#     # Hide any unused subplots
#     for j in range(n, len(axs)):
#         fig.delaxes(axs[j])
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#
#
#     # Display the combined plot
#     plt.show()



def combine_plots(datasets, xlabels, ylabels, titles, parameters, action='show', filename=r'C:\Users\Administrator\Desktop\Ai-lab2\Report\binpack_result',result_counter = 0):

    # Number of plots (plus one for parameters)
    n = len(datasets) + 1

    # Determine the grid size
    cols = 3
    rows = math.ceil(n / cols)

    # Create a figure to hold the subplots
    fig, axs = plt.subplots(rows, cols, figsize=(18, rows * 4))

    # Flatten the array of axes for easy iteration
    axs = axs.flatten()

    # Plot the parameters in the first subplot
    plot_parameters(axs[0], parameters)

    # Create each subplot for the datasets
    for i in range(1, n):
        plot_distribution(axs[i], datasets[i-1], xlabels[i-1], ylabels[i-1], titles[i-1])

    # Hide any unused subplots
    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display or save the combined plot based on the action parameter
    if action == 'show':
        plt.show()
    elif action == 'save':
        counter_str = str(result_counter)
        print("filename: " , filename)
        print("counter str: ", counter_str)
        filename = filename + counter_str + ".png"
        plt.savefig(filename)
    elif action == 'none':
        print("finished the run !")
    else:
        raise ValueError("Action must be either 'show' or 'save'")