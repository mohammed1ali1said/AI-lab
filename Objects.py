from random import random

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

    def __init__(self,grid):
        self.grid = grid
        self.age = 0
        pass

    def init_random_sudoku_individual(self, input_grid):

        # Create a deep copy of the input grid to modify
        new_grid = [row[:] for row in input_grid]

        for row in new_grid:
            # Determine which numbers are already in the row
            existing_numbers = set(row) - {0}
            # Create a list of possible numbers for the row
            possible_numbers = [num for num in range(1, 10) if num not in existing_numbers]

            for i in range(len(row)):
                if row[i] == 0:
                    # Choose a random number from the possible numbers
                    number = random.choice(possible_numbers)
                    row[i] = number
                    # Remove the chosen number from the possible numbers
                    possible_numbers.remove(number)

        self.grid = new_grid


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
