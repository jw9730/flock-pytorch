from scipy.sparse import coo_matrix, csr_array
import numpy as np
import torch
from torch_geometric.data import Data


def adjacency_matrix(data: Data) -> csr_array:
    """Convert edge_index to a sparse adjacency matrix."""
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    assert edge_index is not None
    # Create sparse adjacency matrix from edge_index
    row = edge_index[0].cpu().numpy()
    col = edge_index[1].cpu().numpy()
    data_vals = np.ones(len(row))  # All edges have weight 1    
    A = coo_matrix((data_vals, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    return A


def is_directed(data: Data) -> bool:
    """Fast tensor-based implementation to check if graph is directed."""

    # by pass the check if Data it contains hash
    if hasattr(data, 'is_directed_hash'):
        return data.is_directed_hash

    edge_index = data.edge_index
    assert edge_index is not None
    assert edge_index.dim() == 2 and edge_index.size(0) == 2

    # Create edge hashes to avoid Python loops
    forward_hashes = edge_index[0] * data.num_nodes + edge_index[1]
    reverse_hashes = edge_index[1] * data.num_nodes + edge_index[0]

    # Sort both representations
    forward_sorted = torch.sort(forward_hashes)[0]
    reverse_sorted = torch.sort(reverse_hashes)[0]

    # If undirected, sorted arrays should be identical
    result = not torch.equal(forward_sorted, reverse_sorted)

    data.is_directed_hash = result  # Cache the result in Data

    return result
