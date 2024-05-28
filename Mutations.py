from random import random



def mutate(individual):
    target = list("Hello, world!")
    random_position = random.randint(0, len(target) - 1)  # get a random position for character to mutate
    random_char = chr(random.randint(32, 126))
    individual[random_position] = random_char
    return individual



def inversion_mutation_sudoku(perm,original_row):
    pass

def inversion_mutation_sudoku_grid(grid,originalGrid):
    pass

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
