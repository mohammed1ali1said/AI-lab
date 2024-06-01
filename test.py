import math
import random
# Example Sudoku grid (0 represents empty cells)
input_sudoku_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]


def calc_fittness_sukodu(sudoku_grid,size):
    grid_len = len(sudoku_grid)
    row_score_sum = 0
    col_score_sum = 0
    box_score_sum = 0

    for i in range(0,len(sudoku_grid)):
        row_score_sum += calc_row_fitness(sudoku_grid,i)

    for i in range(0,len(sudoku_grid)):
        col_score_sum += calc_col_fitness(sudoku_grid,i)

    N = len(sudoku_grid)
    step_size = int(math.sqrt(size))

    for i in range(0, N, step_size):
        for j in range(0, N, step_size):
            box_score_sum += calc_box_fitness(sudoku_grid,i,j,int(math.sqrt(size)))

    print("rows fitness :",row_score_sum)
    print("cols fitness :", col_score_sum)
    print("boxes fitness :", box_score_sum)
    return row_score_sum+col_score_sum+box_score_sum
def calc_row_fitness(grid,index): # calculates the fitness of a rowat a certain index in the grid
    row_set = set(grid[index])
    return len(row_set)


def calc_col_fitness(grid,index): # calculates the fitness of a column at a certain index in the grid
    # Extract the specified column
    column = [row[index] for row in grid if len(row) > index]
    # Convert the column to a set to find unique elements
    unique_elements = set(column)

    # Return the number of unique elements
    return len(unique_elements)


def calc_box_fitness(matrix, starting_row_index, starting_column_index, box_row_size): # calculates the fitness of each box of size sqrt(n)Xsqrt(n)

    sub_matrix_elements = []
    # Extract the elements of the sub-matrix
    for i in range(starting_row_index, starting_row_index + box_row_size):
        for j in range(starting_column_index, starting_column_index + box_row_size):
            if i < len(matrix) and j < len(matrix[i]):
                sub_matrix_elements.append(matrix[i][j])

    # Convert the list to a set to find unique elements
    unique_elements = set(sub_matrix_elements)

    # Return the number of unique elements
    return len(unique_elements)


def inversion_mutation_sudoku(perm, original_row):
    # Ensure perm and original_row are valid
    assert len(perm) == 9 and len(original_row) == 9, "Both perm and original_row must be of length 9"

    # Generate two random indices and sort them to ensure idx1 <= idx2
    idx1, idx2 = sorted(random.sample(range(9), 2))

    print("random indexes: ",idx1,idx2)
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


### Test Function

def test_inversion_mutation_sudoku():
    perm = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    original_row = [0, 0, 0, 2, 6, 0, 7, 0, 1]

    print("Original perm:", perm)
    print("Original row:", original_row)

    mutated_perm = inversion_mutation_sudoku(perm, original_row)

    print("Mutated perm:", mutated_perm)


# Run the test
test_inversion_mutation_sudoku()