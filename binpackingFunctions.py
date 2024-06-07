def best_fit(item_sizes, bin_capacities):
    bins = [0] * len(bin_capacities)
    bin_items = [[] for _ in range(len(bin_capacities))]

    for item in item_sizes:
        min_space_left = float('inf')
        best_bin_index = -1

        for i in range(len(bins)):
            space_left = bin_capacities[i] - bins[i]
            if space_left >= item and space_left < min_space_left:
                min_space_left = space_left
                best_bin_index = i

        if best_bin_index != -1:
            bins[best_bin_index] += item
            bin_items[best_bin_index].append(item)
        else:
            print("Item too large to fit in any bin:", item)

    return bin_items


def worst_fit(item_sizes, bin_capacities):
    bins = [0] * len(bin_capacities)
    bin_items = [[] for _ in range(len(bin_capacities))]

    for item in item_sizes:
        max_space_left = -1
        worst_bin_index = -1

        for i in range(len(bins)):
            space_left = bin_capacities[i] - bins[i]
            if space_left >= item and space_left > max_space_left:
                max_space_left = space_left
                worst_bin_index = i

        if worst_bin_index != -1:
            bins[worst_bin_index] += item
            bin_items[worst_bin_index].append(item)
        else:
            print("Item too large to fit in any bin:", item)

    return bin_items

def first_fit(item_sizes, bin_capacities):
    bins = [0] * len(bin_capacities)
    bin_items = [[] for _ in range(len(bin_capacities))]

    for item in item_sizes:
        placed = False
        for i in range(len(bins)):
            if bin_capacities[i] - bins[i] >= item:
                bins[i] += item
                bin_items[i].append(item)
                placed = True
                break

        if not placed:
            print("Item too large to fit in any bin:", item)

    return bin_items

bins = [ ]

