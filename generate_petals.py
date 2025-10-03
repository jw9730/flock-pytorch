import os
import torch
from torch_geometric.data import Data


# Save generated instances to files in the provided directory
def save_instances(ins, path):
    for i, data in enumerate(ins):
        torch.save(data, path + f"_{i}.pth")


# Create a synthetic instance with the given colors, cycle size, and tail length
def create_instance(colourings, cycle_size, tail_len):

    # Extract the number of petals and the maximum color ID
    n_petals = len(colourings)
    max_color =  0
    for pair in colourings:
        max_color = max(max_color, pair[0], pair[1])

    edge_index = []
    edge_type = []

    curr_min = 0

    # Add edges for each petal
    for a, b in colourings:
        edge_index.append([0, curr_min+1])
        edge_type.append(a)
        edge_index.append([0, curr_min+2])
        edge_type.append(b)

        for i in range(0, cycle_size-2):
            edge_index.append([curr_min+2*i + 1, curr_min+2*i + 3])
            edge_type.append(a)
            edge_index.append([curr_min+2*i + 2, curr_min+2*i + 4])
            edge_type.append(a)
        
        edge_index.append([curr_min+2*cycle_size - 2, curr_min+2*cycle_size - 1])
        edge_type.append(a)

        edge_index.append([curr_min+2*cycle_size - 3, curr_min+2*cycle_size  -1])
        edge_type.append(a)

        curr_min += 2*cycle_size - 1
    
    # Add edges for the stem
    edge_index.append([0, curr_min+1])
    edge_type.append(0)
    curr_min += 1
    for i in range(1, tail_len):
        edge_index.append([curr_min, curr_min+1])
        edge_type.append(0)
        curr_min += 1

    # Sample the petal id and position of the target nodes in the petal
    cycle_id = torch.randint(0, n_petals, (1,))
    x = torch.randint(0, cycle_size-1, (1,))

    # Sample the position of the head node in the stem
    tail_pos = torch.randint(0, tail_len, (1,))
    head_id = 0
    if tail_pos > 0:
        head_id = curr_min - tail_pos

    inst = Data(
        edge_index = torch.tensor(edge_index).T,
        edge_type = torch.tensor(edge_type),
        num_nodes = 1 + n_petals * (2*cycle_size-1) + tail_len,
        num_relations = 1 + max_color,
        test_triplets = torch.tensor([
            [head_id, cycle_id * (2*cycle_size-1) + 2*x + 1, 0],
            [head_id, cycle_id * (2*cycle_size-1) + 2*x + 2, 0]
        ]),
    )

    return inst


def main():
    configs = [
        [[1, 2], [2, 1]],
        [[1, 2], [2, 3], [3, 1]],
        [[1, 2], [2, 3], [3, 4], [4, 1]],
        [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1]],
        [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]],
        [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1]],
        [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]],
        [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2], [2, 4], [4, 1], [4, 2], [4, 3], [3, 4], [1,4]],
        [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 1], [4, 1], [4, 2]],
        [[1, 3], [3, 1], [2, 4], [4, 2]],
        [[1, 3], [3, 1], [2, 4], [4, 2], [5, 6], [6, 5]]
    ]

    path = "data/petals/"
    os.makedirs(path, exist_ok=True)
    instances = []

    for colourings in configs:
        for tail_len in [1, 2, 3, 4]:
            for cycle_size in [2, 3, 4, 5, 6]:
                instances.append(create_instance(colourings, cycle_size, tail_len))

    save_instances(instances, path)


if __name__ == "__main__":
    main()
