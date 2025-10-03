from typing import Tuple, Any
import networkx as nx
import numpy as np
from scipy.sparse import diags, csr_array
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from .utils import is_directed, adjacency_matrix


def _weight_node(node: Any, G: nx.Graph, m: int, sub_sampling: float) -> float:
    z = G.degree(node, weight="weight") + 1
    weight = 1 / (z ** sub_sampling)
    return weight


def _weight_node_fast(node: Any, data: Data, m: int, sub_sampling: float) -> float:
    """Fast version of _weight_node for Data with edge_index."""
    # Count both incoming and outgoing edges (total degree)
    out_degree = data.edge_index[0].eq(node).sum().item()
    in_degree = data.edge_index[1].eq(node).sum().item()
    z = out_degree + in_degree + 1  # Add 1 to avoid division by zero
    weight = 1 / (z ** sub_sampling)
    return weight


def get_normalized_adjacency(G: nx.Graph, sub_sampling: float=0.1) -> Tuple[csr_array, np.ndarray]:
    A = nx.adjacency_matrix(G)
    A = A.astype(np.float32)
    probs = A.sum(1)
    if sub_sampling != 0:
        m = len(G.edges)
        D_inv = diags([
            _weight_node(node, G, m, sub_sampling)
            for node in G.nodes
        ])
        A = A.dot(D_inv)

    normalize(A, norm="l1", axis=1, copy=False)
    probs = probs / np.clip(probs.sum(), 1, None)
    return A, probs


def get_normalized_adjacency_fast(data: Data, sub_sampling: float=0.1) -> Tuple[csr_array, np.ndarray]:
    assert data.edge_index is not None and data.num_nodes is not None
    A = adjacency_matrix(data)
    A = A.astype(np.float32)
    probs = A.sum(1)
    if sub_sampling != 0:
        m = len(data.edge_index[0])
        D_inv = diags([
            _weight_node_fast(node, data, m, sub_sampling)
            for node in range(data.num_nodes)
        ])
        A = A.dot(D_inv)

    normalize(A, norm="l1", axis=1, copy=False)
    probs = probs / np.clip(probs.sum(), 1, None)
    return A, probs


def get_normalized_minimum_degree(G: nx.Graph) -> Tuple[csr_array, np.ndarray]:
    """Minimum degree local rule https://arxiv.org/abs/1604.08326"""
    assert not nx.is_directed(G), "Graph must be undirected for minimum degree local rule"
    A = nx.adjacency_matrix(G)
    A = A.astype(np.float32)
    D = A.sum(1)
    A = A.tocoo()
    i, j = A.nonzero()
    A.data = 1 / np.clip(np.minimum(D[i], D[j]), 1, None)
    A = A.tocsr()
    probs = A.sum(1)

    normalize(A, norm="l1", axis=1, copy=False)
    probs = probs / np.clip(probs.sum(), 1, None)
    return A, probs


def get_normalized_minimum_degree_fast(data: Data) -> Tuple[csr_array, np.ndarray]:
    """Minimum degree local rule for Data with edge_index"""
    assert not is_directed(data), "Graph must be undirected for minimum degree local rule"
    A = adjacency_matrix(data)
    A = A.astype(np.float32)
    D = A.sum(1)
    A = A.tocoo()
    i, j = A.nonzero()
    A.data = 1 / np.clip(np.minimum(D[i], D[j]), 1, None)
    A = A.tocsr()
    probs = A.sum(1)

    normalize(A, norm="l1", axis=1, copy=False)
    probs = probs / np.clip(probs.sum(), 1, None)
    return A, probs
