import geneticLab2
import binpacking
# Define the possible choices for parameters
#mutation_controls = ["basic", "non_uniform", "adaptive", "THM", "self_adaptive"]
mutation_controls = ["basic","THM"]
partition_methods = ["crowding", "sharing"]
#partition_methods = ["none"]
mutation_rates = [0.5]

def genetic_algorithm_testing():
    # Define the common parameters
    pop_size = 800
    num_genes = 13
    max_generations = 100
    mutation_rate = 0.5
    crossover_method = "pmx"
    mutation_method = "scramble"
    parent_selection_method = "tournament"
    problem_path = "try1.txt"  # Replace with actual path
    problem = "binpack"
    fitness_func = lambda x: sum(x)  # Replace with actual fitness function
    grid = "easy1"
    show_results = "true"

    # Loop through each combination of mutation_control and partition_method
    counter = 0
    for mutation_control in mutation_controls:
        for partition_method in partition_methods:
            for rate in mutation_rates:
                counter += 1
                # Call the genetic_algorithm function with the current combination of parameters
                geneticLab2.genetic_algorithm(
                    pop_size=pop_size,
                    num_genes=num_genes,
                    max_generations=max_generations,
                    mutation_rate=rate,
                    crossover_method=crossover_method,
                    mutation_method=mutation_method,
                    mutation_control=mutation_control,
                    partition_method=partition_method,
                    parent_selection_method=parent_selection_method,
                    problem_path=problem_path,
                    problem=problem,
                    fitness_func=fitness_func,
                    grid=grid,
                    show_results=show_results,
                    save_result_counter= counter,
                )




#Run the test function
genetic_algorithm_testing()

