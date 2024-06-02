import random
import math

def transpose_matrix(matrix):
    return [list(row) for row in zip(*matrix)]
def mutate(individual):
    target = list("Hello, world!")
    random_position = random.randint(0, len(target) - 1)  # get a random position for character to mutate
    random_char = chr(random.randint(32, 126))
    individual[random_position] = random_char
    return individual



def inversion_mutation_sudoku_row(perm, original_row,size):

    # Ensure perm and original_row are valid
    assert len(perm) == size and len(original_row) == size, "Both perm and original_row must be of length size"

    # Generate two random indices and sort them to ensure idx1 <= idx2
    idx1, idx2 = sorted(random.sample(range(size), 2))


    # Extract the sublist to be potentially reversed
    sublist = perm[idx1:idx2 + 1]

    # Identify elements to be reversed
    elements_to_reverse = [sublist[i - idx1] for i in range(idx1, idx2 + 1) if original_row[i] == 0]

    # Reverse the selected elements
    reversed_elements = elements_to_reverse[::-1]

    # Place the reversed elements back in perm at the appropriate positions
    reverse_index = 0
    for i in range(idx1, idx2 + 1):
        if original_row[i] == 0:
            perm[i] = reversed_elements[reverse_index]
            reverse_index += 1

    return perm

def inversion_mutation_sudoku_grid(grid,originalGrid,size):
    mutated_grid = []
    for row1,row2 in zip(grid,originalGrid):
        mutated_row = inversion_mutation_sudoku_row(row1,row2,size)
        mutated_grid.append(mutated_row)

    return mutated_grid



def find_duplicate_indices(perm):
    duplicates = []
    for i in range(0,len(perm)):
       for j in range(0,len(perm)): # iterate over the indexes of perm
           if(perm[i] == perm[j] and i != j):
               if i not in duplicates:
                    duplicates.append(i)
               if j not in duplicates:
                    duplicates.append(j)

    return duplicates


def find_missing_numbers(perm):
    missing_numbers = []
    for i in range(1,len(perm)+1):
        if i not in perm:
            missing_numbers.append(i)

    return missing_numbers
def inversion_mutation_sudoku_grid_new(grid,originalGrid,size):
    mutated_grid = grid
    for row1,row2 in zip(grid,originalGrid):
        duplicate_indexes = find_duplicate_indices(row1)
        if(len(duplicate_indexes) > 1): # there are duplicates
            for index in duplicate_indexes:
                if row2[index] == 0:
                    column = [row[index] for row in grid]
                    original_column = [row[index] for row in originalGrid]
                    mutated_column = inversion_mutation_sudoku_row(column,original_column,size)
                    for i in range(0,size):
                        mutated_grid[i][index] = mutated_column[i]

    return mutated_grid



def inversion_mutation_sudoku_columns(grid, originalGrid):
    # Transpose the grid to work with columns as rows
    transposed_grid = transpose_matrix(grid)
    transposed_original = transpose_matrix(originalGrid)

    mutated_transposed_grid = []
    for col1, col2 in zip(transposed_grid, transposed_original):
        mutated_col = inversion_mutation_sudoku_row(col1, col2,4)
        mutated_transposed_grid.append(mutated_col)

    # Transpose back to get the final mutated grid
    mutated_grid = transpose_matrix(mutated_transposed_grid)

    return mutated_grid

def block_to_row(block):
    return [num for row in block for num in row]


def row_to_block(lst):
    # Calculate the size of the block (must be a perfect square)
    size = int(math.sqrt(len(lst)))
    assert size * size == len(lst), "List length must be a perfect square"

    # Convert the list to a list of lists (block)
    return [lst[i * size:(i + 1) * size] for i in range(size)]


def extract_column(grid, col_index):
    return [row[col_index] for row in grid]
def extract_3x3_block(grid, start_row, start_col):
    # Extract the 3x3 block
    block = [row[start_col:start_col + 3] for row in grid[start_row:start_row + 3]]
    return block

def extract_random_3x3_block(grid):
    # Generate random indices for the starting point of the 3x3 block
    start_row = random.randint(0, 2) * 3
    start_col = random.randint(0, 2) * 3

    # Extract the 3x3 block
    block = [row[start_col:start_col + 3] for row in grid[start_row:start_row + 3]]

    return start_row,start_col,block

def extract_blocks(grid,size):
    blocks = []
    block_size = int(math.sqrt(size))
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            block = [grid[x][j:j+block_size] for x in range(i, i+block_size)]
            blocks.append(block)
    return blocks


def reconstruct_grid(blocks,size):
    block_size = int(math.sqrt(size))
    grid = [[0]*size for _ in range(size)]
    block_index = 0
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            block = blocks[block_index]
            block_index += 1
            for x in range(block_size):
                grid[i+x][j:j+block_size] = block[x]
    return grid
def scramble_mutation_block(block,original_block):
    block_as_row = block_to_row(block)
    orig_block_as_row = block_to_row(original_block)

    scramble_result_row = scramble_mutation(block_as_row,orig_block_as_row)

    scramble_result_block = row_to_block(scramble_result_row)
    return scramble_result_block


def scramble_mutation_sudoku_grid_block(grid,original_grid,size):

    gridBlocks = extract_blocks(grid,size)
    originalGridBlocks = extract_blocks(original_grid,size)
    crossed_blocks = []
    for block1, block2 in zip(gridBlocks, originalGridBlocks):
        crossed_blocks.append(scramble_mutation_block(block1, block2))

    crossed_grid = reconstruct_grid(crossed_blocks,size)
    return crossed_grid
def scramble_mutation(perm, original_row):

    n = len(original_row)

    for i in range(n - 2):
        if original_row[i] == 0 and original_row[i + 1] == 0 and original_row[i + 2] == 0:
            # Found three adjacent zero cells
            indices = [i, i + 1, i + 2]
            elements_to_scramble = [perm[index] for index in indices]
            random.shuffle(elements_to_scramble)
            for idx, element in zip(indices, elements_to_scramble):
                perm[idx] = element
            break  # Scramble only the first found sequence

    return perm

def scramble_mutation_sudoku_grid(grid,original_grid):
    result = []
    for i in range(0,len(grid)):
        result.append(scramble_mutation(grid[i],original_grid[i]))

    return result


# list_Test = [1,5,4,5,6,7,9,5,2]
# print(find_missing_numbers(list_Test))

# sudoku_grid = [
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [0, 9, 8, 5, 0, 0, 0, 6, 0],
#     [8, 0, 0, 7, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 9, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ]
#
# sudoku_grid_original = [
#     [5, 3, 8, 6, 7, 0, 0, 0, 0],
#     [6, 0, 3, 1, 9, 5, 0, 4, 0],
#     [0, 9, 8, 0, 0, 0, 0, 6, 1],
#     [8, 0, 2, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 0, 5, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 7, 2, 8, 0],
#     [0, 0, 0, 4, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 8, 0, 0, 7, 9]
# ]
#
# row,col,random_block = extract_random_3x3_block(sudoku_grid)
# roww = row
# coll = col
# print("coordinate for chosen block: ", (row,col))
# # print("Random 3x3 block:")
# # for row in random_block:
# #     print(row)
#
#
# block = extract_3x3_block(sudoku_grid,row,col)
#
# mutated_block = scramble_mutation_block(random_block,block)
# for i in range(0,len(mutated_block)):
#     for j in range(0,len(mutated_block)):
#         mutated_block[i][j] += 1
# print("block is:")
# for row in block:
#     print(row)
#
# print("mutated block:")
# for row in mutated_block:
#     print(row)
#
# for i in range(0, 3):
#     for j in range(0, 3):
#         sudoku_grid[roww + i][coll + j] = mutated_block[i][j]
#
# for row in sudoku_grid:
#     print(row)


# random_col_index = random.randint(0, 8)
# print("random col index: ", random_col_index)
# extracted_column = extract_column(sudoku_grid, 3)
# print("extracted col: ",extracted_column)
# original_column = extract_column(sudoku_grid_original, 3)
# print("original col: ",original_column)
# mutated_column = scramble_mutation(extracted_column, original_column)
# print("mutated col:",mutated_column)
# for i in range(0, len(sudoku_grid)):
#     sudoku_grid[i][3] = mutated_column[i]
#
# for row in sudoku_grid:
#     print(row)