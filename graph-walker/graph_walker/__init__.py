# pylint: disable=no-name-in-module,import-error
import time
import numpy as np
import networkx as nx
from torch_geometric.data import Data

from _walker import random_walks as _random_walks
from _walker import random_walks_with_restart as _random_walks_with_restart
from _walker import random_walks_with_no_backtrack as _random_walks_with_no_backtrack
from _walker import random_walks_with_restart_no_backtrack as _random_walks_with_restart_no_backtrack
from _walker import random_walks_with_periodic_restart as _random_walks_with_periodic_restart
from _walker import random_walks_with_periodic_restart_no_backtrack as _random_walks_with_periodic_restart_no_backtrack
from _walker import node2vec_random_walks as _node2vec_random_walks

from _walker import anonymize as _anonymize
from _walker import anonymize_edge_types as _anonymize_edge_types
from _walker import anonymize_with_neighbors as _anonymize_with_neighbors
from _walker import parse_edge_types_and_directions as _parse_edge_types_and_directions
from _walker import parse_edge_types_and_directions_with_neighbors as _parse_edge_types_and_directions_with_neighbors

from _walker import as_text as _as_text
from _walker import as_text_with_neighbors as _as_text_with_neighbors


from .preprocessing import get_normalized_adjacency, get_normalized_minimum_degree, get_normalized_adjacency_fast, get_normalized_minimum_degree_fast
from .utils import is_directed, adjacency_matrix


def transition_probs(graph: nx.Graph, min_degree=False, sub_sampling=0.):
    assert not nx.is_directed(graph), "Graph must be undirected"
    if min_degree:
        A, _ = get_normalized_minimum_degree(graph)
    else:
        A, _ = get_normalized_adjacency(graph, sub_sampling=sub_sampling)

    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)
    return indptr, indices, data


def transition_probs_fast(graph: Data, min_degree=False, sub_sampling=0.):
    """Compute transition probabilities for a graph represented by edge_index.

    Args:
        graph (torch_geometric.data.Data): Graph data containing edge_index, containing the edge indices and num_nodes.
            edge_index (torch.Tensor): Edge index of the graph with shape [2, num_edges]
        min_degree (bool): Whether to use minimum degree local rule.
        sub_sampling (float): Subsampling parameter for normalized adjacency.

    Returns:
        indptr (np.ndarray): Indices for the start of each row in the sparse matrix.
        indices (np.ndarray): Column indices of the non-zero entries.
        data (np.ndarray): Values of the non-zero entries.
    """
    edge_index = graph.edge_index
    assert edge_index is not None
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, "edge_index must be a 2D tensor with shape [2, num_edges]"
    assert not is_directed(graph), "Graph must be undirected"

    if min_degree:
        A, _ = get_normalized_minimum_degree_fast(graph)
    else:
        A, _ = get_normalized_adjacency_fast(graph, sub_sampling=sub_sampling)

    indptr = A.indptr.astype(np.uint32)
    indices = A.indices.astype(np.uint32)
    data = A.data.astype(np.float32)
    return indptr, indices, data


def _prefix(graph: nx.Graph, prefix, restarts):
    if prefix is None:
        prefix = np.expand_dims(np.arange(len(graph.nodes)).astype(np.uint32), axis=1)
    else:
        assert prefix.data.contiguous, "Prefix must be contiguous"
        prefix = np.array(prefix, dtype=np.uint32)
        if prefix.ndim == 1:
            prefix = np.expand_dims(prefix, axis=1)
    if restarts is None:
        restarts = np.zeros((prefix.shape[0], 1), dtype=bool)
    else:
        assert restarts.data.contiguous, "Restarts must be contiguous"
        restarts = np.array(restarts, dtype=bool)
        if restarts.ndim == 1:
            restarts = np.expand_dims(restarts, axis=1)
    return prefix, restarts


def _prefix_fast(graph: Data, prefix, restarts):
    if prefix is None:
        assert graph.num_nodes is not None
        prefix = np.expand_dims(np.arange(graph.num_nodes).astype(np.uint32), axis=1)
    else:
        assert prefix.data.contiguous, "Prefix must be contiguous"
        prefix = np.array(prefix, dtype=np.uint32)
        if prefix.ndim == 1:
            prefix = np.expand_dims(prefix, axis=1)
    if restarts is None:
        restarts = np.zeros((prefix.shape[0], 1), dtype=bool)
    else:
        assert restarts.data.contiguous, "Restarts must be contiguous"
        restarts = np.array(restarts, dtype=bool)
        if restarts.ndim == 1:
            restarts = np.expand_dims(restarts, axis=1)
    return prefix, restarts


def _seed(seed):
    if seed is None:
        seed = int(np.random.rand() * (2**32 - 1))
    return seed


def random_walks(
    graph: nx.Graph,
    n_walks=10,
    walk_len=10,
    min_degree=False,
    sub_sampling=0.,
    p=1, q=1, alpha=0, k=None,
    no_backtrack=False,
    prefix=None,
    prefix_restarts=None,
    seed=None,
    verbose=True
):
    """Generate random walks on a graph.

    Args:
        graph (nx.Graph): Graph to walk on.
        n_walks (int): Number of walks per node.
        walk_len (int): Length of each walk.
        min_degree (bool): Whether to use minimum degree local rule.
        sub_sampling (float): Subsampling parameter for normalized adjacency.
        p (float): Return parameter.
        q (float): In-out parameter.
        alpha (float): Restart probability.
        k (int): Restart period.
        no_backtrack (bool): Whether to disallow backtracking.
        prefix (list): List of nodes to start walks from, or list of node sequences to resume walks from.
        prefix_restarts (list): List of boolean sequences indicating whether each prefix step is a restart.
        seed (int): Random seed.
        verbose (bool): Whether to print progress.

    Returns:
        walks (np.ndarray): Random walk matrix of shape (n_start_nodes * n_walks, walk_len).
        restarts (np.ndarray): Restart matrix of shape (n_start_nodes * n_walks, walk_len).
    """
    start_time = time.time()

    indptr, indices, data = transition_probs(graph, min_degree, sub_sampling)
    prefix, prefix_restarts = _prefix(graph, prefix, prefix_restarts)
    seed = _seed(seed)

    if p == 1 and q == 1:
        if alpha == 0:
            if k is None:
                if no_backtrack:
                    walks = _random_walks_with_no_backtrack(indptr, indices, data, prefix, seed, n_walks, walk_len)
                else:
                    walks = _random_walks(indptr, indices, data, prefix, seed, n_walks, walk_len)
                restarts = np.zeros(walks.shape, dtype=bool)
            else:
                if no_backtrack:
                    walks = _random_walks_with_periodic_restart_no_backtrack(indptr, indices, data, prefix, seed, n_walks, walk_len, k)
                else:
                    walks = _random_walks_with_periodic_restart(indptr, indices, data, prefix, seed, n_walks, walk_len, k)
                restarts = np.zeros(walks.shape, dtype=bool)
                restarts[:, k::k] = True
        else:
            assert k is None, "Periodic restarts are not implemented for randomly restarting walks"
            if no_backtrack:
                walks, restarts = _random_walks_with_restart_no_backtrack(indptr, indices, data, prefix, prefix_restarts, seed, n_walks, walk_len, alpha)
            else:
                walks, restarts = _random_walks_with_restart(indptr, indices, data, prefix, prefix_restarts, seed, n_walks, walk_len, alpha)
    else:
        assert alpha == 0, "Restarts are not implemented for node2vec walks"
        assert k is None, "Periodic restarts are not implemented for node2vec walks"
        if no_backtrack:
            raise NotImplementedError("Non-backtracking is not implemented for node2vec walks")
        walks = _node2vec_random_walks(indptr, indices, data, prefix, seed, n_walks, walk_len, p, q)
        restarts = np.zeros(walks.shape, dtype=bool)

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")

    return walks, restarts


def random_walks_fast(
    graph: Data,
    n_walks=10,
    walk_len=10,
    min_degree=False,
    sub_sampling=0.,
    p=1, q=1, alpha=0, k=None,
    no_backtrack=False,
    prefix=None,
    prefix_restarts=None,
    seed=None,
    verbose=True
):
    start_time = time.time()

    indptr, indices, data = transition_probs_fast(graph, min_degree, sub_sampling)
    prefix, prefix_restarts = _prefix_fast(graph, prefix, prefix_restarts)
    seed = _seed(seed)

    if p == 1 and q == 1:
        if alpha == 0:
            if k is None:
                if no_backtrack:
                    walks = _random_walks_with_no_backtrack(indptr, indices, data, prefix, seed, n_walks, walk_len)
                else:
                    walks = _random_walks(indptr, indices, data, prefix, seed, n_walks, walk_len)
                restarts = np.zeros(walks.shape, dtype=bool)
            else:
                if no_backtrack:
                    walks = _random_walks_with_periodic_restart_no_backtrack(indptr, indices, data, prefix, seed, n_walks, walk_len, k)
                else:
                    walks = _random_walks_with_periodic_restart(indptr, indices, data, prefix, seed, n_walks, walk_len, k)
                restarts = np.zeros(walks.shape, dtype=bool)
                restarts[:, k::k] = True
        else:
            assert k is None, "Periodic restarts are not implemented for randomly restarting walks"
            if no_backtrack:
                walks, restarts = _random_walks_with_restart_no_backtrack(indptr, indices, data, prefix, prefix_restarts, seed, n_walks, walk_len, alpha)
            else:
                walks, restarts = _random_walks_with_restart(indptr, indices, data, prefix, prefix_restarts, seed, n_walks, walk_len, alpha)
    else:
        assert alpha == 0, "Restarts are not implemented for node2vec walks"
        assert k is None, "Periodic restarts are not implemented for node2vec walks"
        if no_backtrack:
            raise NotImplementedError("Non-backtracking is not implemented for node2vec walks")
        walks = _node2vec_random_walks(indptr, indices, data, prefix, seed, n_walks, walk_len, p, q)
        restarts = np.zeros(walks.shape, dtype=bool)

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")

    return walks, restarts


def as_text(walks, restarts, graph: nx.Graph, include_neighbors=True, verbose=True):
    """Convert random walks to text strings.

    Args:
        walks (np.ndarray): Random walk matrix of shape (n_start_nodes * n_walks, walk_len).
        restarts (np.ndarray): Restart matrix of shape (n_start_nodes * n_walks, walk_len).
        graph (nx.Graph): Graph to convert walks from.
        include_neighbors (bool): Whether to include neighbors in the text.
        verbose (bool): Whether to print progress.

    Returns:
        walks_text (list): Stringified random walk list of length n_start_nodes * n_walks.
    """
    start_time = time.time()

    if include_neighbors:
        assert not nx.is_directed(graph), "Graph must be undirected"
        A = nx.adjacency_matrix(graph)
        indptr = A.indptr.astype(np.uint32)
        indices = A.indices.astype(np.uint32)
        named_walks, walks, restarts, neighbors = _anonymize_with_neighbors(walks, restarts, indptr, indices)
        walks_text = _as_text_with_neighbors(named_walks, restarts, neighbors)
    else:
        named_walks = _anonymize(walks)
        walks_text = _as_text(named_walks, restarts)

    if verbose:
        duration = time.time() - start_time
        print(f"Text conversion - T={duration:.2f}s")

    return walks_text


def as_text_fast(walks, restarts, graph: Data, include_neighbors=True, verbose=True):
    start_time = time.time()

    if include_neighbors:
        assert not is_directed(graph), "Graph must be undirected"
        A = adjacency_matrix(graph)
        indptr = A.indptr.astype(np.uint32)
        indices = A.indices.astype(np.uint32)
        named_walks, walks, restarts, neighbors = _anonymize_with_neighbors(walks, restarts, indptr, indices)
        walks_text = _as_text_with_neighbors(named_walks, restarts, neighbors)
    else:
        named_walks = _anonymize(walks)
        walks_text = _as_text(named_walks, restarts)

    if verbose:
        duration = time.time() - start_time
        print(f"Text conversion - T={duration:.2f}s")

    return walks_text


def random_walks_with_precomputed_probs(
    indptr, indices, data,
    n_walks=10,
    walk_len=10,
    p=1, q=1, alpha=0, k=None,
    no_backtrack=False,
    prefix=None,
    prefix_restarts=None,
    seed=None,
    verbose=True
):
    start_time = time.time()

    prefix = np.array(prefix, dtype=np.uint32)
    if prefix.ndim == 1:
        prefix = np.expand_dims(prefix, axis=1)
    if prefix_restarts is None:
        prefix_restarts = np.zeros((prefix.shape[0], 1), dtype=bool)
    else:
        prefix_restarts = np.array(prefix_restarts, dtype=bool)
    seed = _seed(seed)

    if p == 1 and q == 1:
        if alpha == 0:
            if k is None:
                if no_backtrack:
                    walks = _random_walks_with_no_backtrack(indptr, indices, data, prefix, seed, n_walks, walk_len)
                else:
                    walks = _random_walks(indptr, indices, data, prefix, seed, n_walks, walk_len)
                restarts = np.zeros(walks.shape, dtype=bool)
            else:
                if no_backtrack:
                    walks = _random_walks_with_periodic_restart_no_backtrack(indptr, indices, data, prefix, seed, n_walks, walk_len, k)
                else:
                    walks = _random_walks_with_periodic_restart(indptr, indices, data, prefix, seed, n_walks, walk_len, k)
                restarts = np.zeros(walks.shape, dtype=bool)
                restarts[:, k::k] = True
        else:
            assert k is None, "Periodic restarts are not implemented for randomly restarting walks"
            if no_backtrack:
                walks, restarts = _random_walks_with_restart_no_backtrack(indptr, indices, data, prefix, prefix_restarts, seed, n_walks, walk_len, alpha)
            else:
                walks, restarts = _random_walks_with_restart(indptr, indices, data, prefix, prefix_restarts, seed, n_walks, walk_len, alpha)
    else:
        assert alpha == 0, "Restarts are not implemented for node2vec walks"
        assert k is None, "Periodic restarts are not implemented for node2vec walks"
        if no_backtrack:
            raise NotImplementedError("Non-backtracking is not implemented for node2vec walks")
        walks = _node2vec_random_walks(indptr, indices, data, prefix, seed, n_walks, walk_len, p, q)
        restarts = np.zeros(walks.shape, dtype=bool)

    if verbose:
        duration = time.time() - start_time
        print(f"Random walks - T={duration:.2f}s")

    return walks, restarts


def stationary_distribution(graph: nx.Graph, min_degree=False, sub_sampling=0.):
    """Compute the stationary distribution of a graph.

    Args:
        graph (nx.Graph): Graph to compute stationary distribution for.
        min_degree (bool): Whether to use minimum degree local rule.
        sub_sampling (float): Subsampling parameter for normalized adjacency.

    Returns:
        probs (np.ndarray): Stationary distribution of shape (n_nodes,).
    """
    assert not nx.is_directed(graph), "Graph must be undirected"
    if min_degree:
        _, probs = get_normalized_minimum_degree(graph)
    else:
        _, probs = get_normalized_adjacency(graph, sub_sampling=sub_sampling)
    return probs


def stationary_distribution_fast(graph: Data, min_degree=False, sub_sampling=0.):
    assert is_directed(graph), "Graph must be undirected"
    if min_degree:
        _, probs = get_normalized_minimum_degree_fast(graph)
    else:
        _, probs = get_normalized_adjacency_fast(graph, sub_sampling=sub_sampling)
    return probs
