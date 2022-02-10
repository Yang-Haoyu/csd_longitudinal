"""
Generate the DAG. Some of the functions are from https://github.com/xunzheng/notears
"""

import networkx as nx
import numpy as np


def _random_permutation(rng, M):
    # np.random.permutation permutes first axis only
    P = rng.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P


def _random_acyclic_orientation(rng, B_und):
    return np.tril(_random_permutation(rng, B_und), k=-1)


def gen_dag(rng, seed, N_node=10, density=0.4):
    G = nx.gnp_random_graph(N_node, density, seed=seed, directed=True)
    B_und = nx.adjacency_matrix(G).todense()
    B = _random_acyclic_orientation(rng, B_und)
    G = nx.from_numpy_matrix(B, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)
    return B, G
