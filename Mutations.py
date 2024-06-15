import random
import math


def transpose_matrix(matrix): # takes a matrix and transposes it
    return [list(row) for row in zip(*matrix)]
def mutate(individual): # takes an individual and mutates it. (this is for the string individual on the string problem)
    target = list("Hello, world!")
    random_position = random.randint(0, len(target) - 1)  # get a random position for character to mutate
    random_char = chr(random.randint(32, 126))
    individual[random_position] = random_char
    return individual



def inversion_mutation_sudoku_row(perm, original_row,size): # takes a row, and the original row in the input matrix. takes also the size of the row. and applies inversion mutation on the row.

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

def inversion_mutation_sudoku_grid(grid,originalGrid,size): # takes a grid, and the original input grid, and the size. and applies inversion mutation on all the grid.
    mutated_grid = []
    for row1,row2 in zip(grid,originalGrid): # apply inversion on each row
        mutated_row = inversion_mutation_sudoku_row(row1,row2,size)
        mutated_grid.append(mutated_row)

    return mutated_grid



def find_duplicate_indices(perm): # takes a row (permutation) and returns a list of indexes of equal elements.
    duplicates = []
    for i in range(0,len(perm)):
       for j in range(0,len(perm)): # iterate over the indexes of perm
           if(perm[i] == perm[j] and i != j):
               if i not in duplicates:
                    duplicates.append(i)
               if j not in duplicates:
                    duplicates.append(j)

    return duplicates


def find_missing_numbers(perm): # takes a permutation and finds the missing numbers, from 1 to the length of the permutation
    missing_numbers = []
    for i in range(1,len(perm)+1):
        if i not in perm:
            missing_numbers.append(i)

    return missing_numbers
def inversion_mutation_sudoku_grid_new(grid,originalGrid,size): # this function applies inversion mutation only on columns that contain duplicates
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



def inversion_mutation_sudoku_columns(grid, originalGrid): # applies an inversion mutation on the given grid, but the inversions are made on columns and not rows
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

def block_to_row(block): # changes a given block to a row
    return [num for row in block for num in row]


def row_to_block(lst): # changes a given row to a bock. for example it takes a permutation of size 9 (list of size 9) . and makes it a block of 3x3
    # Calculate the size of the block (must be a perfect square)
    size = int(math.sqrt(len(lst)))
    assert size * size == len(lst), "List length must be a perfect square"

    # Convert the list to a list of lists (block)
    return [lst[i * size:(i + 1) * size] for i in range(size)]


def extract_column(grid, col_index): # given a grid extract the column at given index
    return [row[col_index] for row in grid]
def extract_3x3_block(grid, start_row, start_col): # given a grid extract a block at the given coordinates.
    # Extract the 3x3 block
    block = [row[start_col:start_col + 3] for row in grid[start_row:start_row + 3]]
    return block

def extract_random_3x3_block(grid): # extracts a random 3x3 block from a sudoku grid.
    # Generate random indices for the starting point of the 3x3 block
    start_row = random.randint(0, 2) * 3
    start_col = random.randint(0, 2) * 3

    # Extract the 3x3 block
    block = [row[start_col:start_col + 3] for row in grid[start_row:start_row + 3]]

    return start_row,start_col,block

def extract_blocks(grid,size): # extracts all blocks from a grid
    blocks = []
    block_size = int(math.sqrt(size))
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            block = [grid[x][j:j+block_size] for x in range(i, i+block_size)]
            blocks.append(block)
    return blocks


def reconstruct_grid(blocks,size): # reconstructs a grid from given blocks
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
def scramble_mutation_block(block,original_block): # takes a block and applies scramble mutation on it
    block_as_row = block_to_row(block)
    orig_block_as_row = block_to_row(original_block)

    scramble_result_row = scramble_mutation(block_as_row,orig_block_as_row)

    scramble_result_block = row_to_block(scramble_result_row)
    return scramble_result_block


def scramble_mutation_sudoku_grid_block(grid,original_grid,size): # takes a grid and applies scramble mutation on each block of 3x3

    gridBlocks = extract_blocks(grid,size)
    originalGridBlocks = extract_blocks(original_grid,size)
    crossed_blocks = []
    for block1, block2 in zip(gridBlocks, originalGridBlocks):
        crossed_blocks.append(scramble_mutation_block(block1, block2))

    crossed_grid = reconstruct_grid(crossed_blocks,size)
    return crossed_grid
def scramble_mutation(perm, original_row): # takes a permutation and applies scramble mutation

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

def scramble_mutation_sudoku_grid(grid,original_grid): # takes a grid and applies scramble mutation on each row
    result = []
    for i in range(0,len(grid)):
        result.append(scramble_mutation(grid[i],original_grid[i]))

    return result


def replacement_mutation(grid,original_grid): # takes a grid, and applies replacement mutation on each row

    for i in  range(0,len(grid)):
        random_idx = random.randint(0,8)

        while(original_grid[i][random_idx] == 0):
            random_idx = random.randint(0, 8)


        random_num = random.randint(1,9)
        grid[i][random_idx] = random_num
    return grid



def apply_thm_test(thm_best_fit_tracker,thm_thresh):
    count = 0
    for i in range(len(thm_best_fit_tracker) - 1):
        if thm_best_fit_tracker[i] < thm_best_fit_tracker[i + 1]:
            count += 1

    if count/len(thm_best_fit_tracker) < thm_thresh :
        print("the THM test passed, we will increase mutation rate now")
        return "true"
    else:
        print("the THM test failed, it seems that the enhancement progress is healthy!")
        return "false"
