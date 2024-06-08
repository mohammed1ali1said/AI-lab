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

class StringIndividual:
      def __init__(self,num_genes):
          self.str = [chr(random.randint(32, 126)) for j in range(num_genes)]
          self.age = 0


class SudokuIndividual:

    def __init__(self,grid,size):
        self.size = size
        self.grid = grid
        self.age = 0
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





# def plot_distribution(ax, data, xlabel, ylabel, title):
#     """Creates a plot on the given axis."""
#     ax.plot(data, marker='none', linestyle='-', color='b')
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#
# def plot_parameters(ax, parameters):
#     """Creates a text plot for the given parameters on the given axis."""
#     param_text = "\n".join([f"{key}: {value}" for key, value in parameters.items()])
#     ax.text(0.5, 0.5, param_text, fontsize=12, ha='center', va='center', wrap=True)
#     ax.axis('off')
#     ax.set_title('Algorithm Parameters')
#
#
#
#
# def combine_plots(datasets, xlabels, ylabels, titles, parameters, sudoku_grid=None):
#     """Combines multiple plots into a single image, including parameters and optionally a Sudoku grid."""
#     # Number of plots (plus one for parameters)
#     n = len(datasets) + 1
#     if sudoku_grid is not None:
#         n += 1
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
#     for i in range(1, len(datasets) + 1):
#         plot_distribution(axs[i], datasets[i-1], xlabels[i-1], ylabels[i-1], titles[i-1])
#
#     # Plot the Sudoku grid if provided
#     if sudoku_grid is not None:
#         plot_sudoku(axs[len(datasets) + 1], sudoku_grid)
#
#
#     # Hide any unused subplots
#     for j in range(n, len(axs)):
#         fig.delaxes(axs[j])
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#
#     # Display the combined plot
#     plt.show()
#
def plot_sudoku(ax, grid):
    """Plots a Sudoku grid using matplotlib."""
    ax.axis('off')
    table = ax.table(cellText=grid, cellLoc='center', loc='center', colWidths=[0.1]*len(grid))
    table.scale(1, 1.5)

    # Style the table
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            cell = table[(i, j)]
            cell.set_fontsize(14)
            if grid[i][j] == 0:  # Highlight empty cells (optional)
                cell.set_facecolor("#f3f3f3")
            else:
                cell.set_facecolor("#ffffff")
            cell.set_edgecolor("black")
    ax.set_title('Best Individual')


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

def plot_results(ax, results):
    """Creates a text plot for the algorithm results on the given axis."""
    ax.text(0.5, 0.5, results, fontsize=12, ha='center', va='center', wrap=True)
    ax.axis('off')
    ax.set_title('Algorithm Results')

def combine_plots(datasets, xlabels, ylabels, titles, parameters, results, sudoku_grid=None):
    """Combines multiple plots into a single image, including parameters, results, and optionally a Sudoku grid."""
    # Number of plots (plus one for parameters and one for results)
    n = len(datasets) + 2
    if sudoku_grid is not None:
        n += 1

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
    for i in range(1, len(datasets) + 1):
        plot_distribution(axs[i], datasets[i-1], xlabels[i-1], ylabels[i-1], titles[i-1])

    # Plot the Sudoku grid if provided
    if sudoku_grid is not None:
        plot_sudoku(axs[len(datasets) + 1], sudoku_grid)

    # Plot the results in the next subplot
    plot_results(axs[len(datasets) + (1 if sudoku_grid is None else 2)], results)

    # Hide any unused subplots
    for j in range(n, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the combined plot
    plt.show()