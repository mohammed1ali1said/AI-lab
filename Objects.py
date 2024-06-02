import copy
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

def plot_distribution(data, xlabel, ylabel, title):
    # Create a histogram
        plt.figure(figsize=(10, 6))
        plt.plot(data, marker='none', linestyle='-', color='b')

        # Adding labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()


