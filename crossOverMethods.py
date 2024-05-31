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


def pmx_crossover_sudoku(parent1, parent2,
                         original_row):  # applies pmx crossover on 2 given permutations (2 rows in the gird in our case)
    # Get a random position

    random_index = random.randint(0, len(parent1) - 1)
    while original_row[
        random_index] != 0:  # make sure that the index we are swapping was not in the original input matrix, (because otherwsie we are not solving the given problem)
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


def extract_blocks(grid):
    blocks = []
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = [grid[x][j:j+3] for x in range(i, i+3)]
            blocks.append(block)
    return blocks


def reconstruct_grid(blocks):
    grid = [[0]*9 for _ in range(9)]
    block_index = 0
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block = blocks[block_index]
            block_index += 1
            for x in range(3):
                grid[i+x][j:j+3] = block[x]
    return grid



def pmx_crossover_sudoku_grid_block(parent1:objects.SudokuIndividual,parent2:objects.SudokuIndividual,original_grid):
    # Note : this function return 1 child grid, but its implementation supports returning 2 childs in case we needed this later
    firstParent = parent1.grid
    secondParent = parent2.grid
    offspring1 = []
    offspring2 = []

    firstParentBlocks = extract_blocks(firstParent)
    secondParentBlocks = extract_blocks(secondParent)
    originalGridBlocks = extract_blocks(original_grid)
    crossed_blocks = []
    for block1, block2,orig_block in zip(firstParentBlocks, secondParentBlocks,originalGridBlocks):
        crossed_blocks.append(pmx_crossover_sudoku_block(block1,block2,orig_block))

    crossed_grid = reconstruct_grid(crossed_blocks)
    return crossed_grid
def cx_crossover_sudoku(parent1, parent2,original_row): # applies cx crossover on 2 given permutations (2 rows in the gird in our case)

    pass



# block1_test = [[1,2,4],[3,4,5],[6,7,8]]
# block2_test = [[2,3,5],[4,5,6],[7,8,9]]
# original_block = [[1,0,0],[0,4,0],[6,0,8]]
#
# result1,result2 = pmx_crossover_sudoku_block(block1_test,block2_test,original_block)
# for row in block1_test:
#     print(row)
# print("---------------------------")
# for row in result1:
#     print(row)
#
#
# print(" ")
# for row in block2_test:
#     print(row)
# print("---------------------------")
# for row in result2:
#     print(row)


