import random
import  Objects as objects

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

def pmx_crossover_sudoku(parent1, parent2,original_row): # applies pmx crossover on 2 given permutations (2 rows in the gird in our case)
        # Get a random position

        random_index = random.randint(0, len(parent1) - 1)
        while original_row[random_index] != 0: # make sure that the index we are swapping was not in the original input matrix, (because otherwsie we are not solving the given problem)
            random_index = random.randint(0, len(parent1) - 1)


        temp = parent1[random_index]
        parent1_copy = list(parent1)
        parent2_copy = list(parent2)
        # Swap the cells from parent 1 and parent 2
        parent1_copy[random_index] = parent2_copy[random_index]
        parent2_copy[random_index] = temp

        # remove the old duplicate
        for i in range(0,len(parent1_copy)):
            if i != random_index and parent1_copy[i] == parent1_copy[random_index]:
                parent1_copy[i] = parent2_copy[random_index]

            if i != random_index and parent2_copy[i] == parent2_copy[random_index]:
                parent2_copy[i] = parent1_copy[random_index]

        return parent1_copy,parent2_copy

def pmx_crossover_sudoku_grid(parent1:objects.SudokuIndividual,parent2:objects.SudokuIndividual,original_grid): # this function applies the pmx crossover on 2 parent grids and returns 1 child grid
        # Note : this function return 1 child grid, but its implementation supports returning 2 childs in case we needed this later
        firstParent = parent1.grid
        secondParent = parent2.grid
        offspring1 = []
        offspring2 = []
        for i in range(0, len(firstParent)):
            res1, res2 = pmx_crossover_sudoku(firstParent[i], secondParent[i], original_grid[i])
            offspring1.append(res1)
            offspring2.append(res2)
        return offspring1


def cx_crossover_sudoku(parent1, parent2,original_row): # applies cx crossover on 2 given permutations (2 rows in the gird in our case)

    pass


