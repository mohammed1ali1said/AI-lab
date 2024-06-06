import random
import math
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


def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]


def block_to_row(block):
    return [num for row in block for num in row]


def row_to_block(lst):
    # Calculate the size of the block (must be a perfect square)
    size = int(math.sqrt(len(lst)))
    assert size * size == len(lst), "List length must be a perfect square"

    # Convert the list to a list of lists (block)
    return [lst[i * size:(i + 1) * size] for i in range(size)]
def pmx_crossover_sudoku_block(parent1,parent2,original_block):

    parent1_as_row = block_to_row(parent1)
    parent2_as_row = block_to_row(parent2)
    original_block_as_row = block_to_row(original_block)

    res1,res2 = pmx_crossover_sudoku(parent1_as_row,parent2_as_row,original_block_as_row)
    res1_block = row_to_block(res1)
    res2_block = row_to_block(res2)

    return res1_block


def row_contains_zeros(row):
    for num in row:
        if num == 0:
            return True
    return False

def pmx_crossover_sudoku(parent1, parent2,
                         original_row):  # applies pmx crossover on 2 given permutations (2 rows in the gird in our case)
    # Get a random position

    random_index = random.randint(0, len(parent1) - 1)


    if(row_contains_zeros(original_row)):
        while original_row[random_index] != 0:  # make sure that the index we are swapping was not in the original input matrix, (because otherwsie we are not solving the given problem)
            random_index = random.randint(0, len(parent1) - 1)


    temp = parent1[random_index]
    parent1_copy = list(parent1)
    parent2_copy = list(parent2)
    # Swap the cells from parent 1 and parent 2
    parent1_copy[random_index] = parent2_copy[random_index]
    parent2_copy[random_index] = temp

    # remove the old duplicate
    for i in range(0, len(parent1_copy)):
        if i != random_index and parent1_copy[i] == parent1_copy[random_index] and original_row[i] == 0:
            parent1_copy[i] = parent2_copy[random_index]

        if i != random_index and parent2_copy[i] == parent2_copy[random_index] and original_row[i] == 0:
            parent2_copy[i] = parent1_copy[random_index]

    return parent1_copy, parent2_copy



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


def extract_blocks(grid,size):
    blocks = []
    block_size = math.sqrt(size)
    for i in range(0, size, int(block_size)):
        for j in range(0, size, int(block_size)):
            block = [grid[x][j:j+int(block_size)] for x in range(i, i+int(block_size))]
            blocks.append(block)
    return blocks


def reconstruct_grid(blocks,size):
    block_size = int(math.sqrt(size))
    grid = [[0]*size for _ in range(size)]
    block_index = 0
    for i in range(0, size, int(block_size)):
        for j in range(0, size, int(block_size)):
            block = blocks[block_index]
            block_index += 1
            for x in range(int(block_size)):
                grid[i+x][j:j+int(block_size)] = block[x]
    return grid


def pmx_crossover_sudoku_grid_block(parent1:objects.SudokuIndividual,parent2:objects.SudokuIndividual,original_grid):
    # Note : this function return 1 child grid, but its implementation supports returning 2 childs in case we needed this later
    firstParent = parent1.grid
    secondParent = parent2.grid


    offspring1 = []
    offspring2 = []


    firstParentBlocks = extract_blocks(firstParent,len(firstParent))
    secondParentBlocks = extract_blocks(secondParent,len(firstParent))
    originalGridBlocks = extract_blocks(original_grid,len(firstParent))

    crossed_blocks = []

    for block1, block2,orig_block in zip(firstParentBlocks, secondParentBlocks,originalGridBlocks):
        crossed_blocks.append(pmx_crossover_sudoku_block(block1,block2,orig_block))

    crossed_grid = reconstruct_grid(crossed_blocks,len(firstParent))
    return crossed_grid



def find_cycles(parent1, parent2): # returns a list of cycles where each cycle is represented as a list of indices
    """Finds cycles in two parent permutations."""
    cycles = []
    visited = [False] * len(parent1)
    for i in range(len(parent1)):
        if not visited[i]:
            cycle = []
            x = i
            while not visited[x]:
                cycle.append(x)
                visited[x] = True
                x = parent1.index(parent2[x])
            cycles.append(cycle)
    return cycles


def cx_crossover(parent1, parent2):
    """Performs cycle crossover on two parent permutations."""
    assert len(parent1) == len(parent2), "Permutations must be of the same length"

    offspring1 = parent1[:]
    offspring2 = parent2[:]

    cycles = find_cycles(parent1, parent2)

    for i, cycle in enumerate(cycles):
        if i % 2 == 0:  # Swap the elements of the cycle in even-indexed cycles
            for index in cycle:
                offspring1[index], offspring2[index] = offspring2[index], offspring1[index]

    return offspring1, offspring2





