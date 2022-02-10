from gen_dag import gen_dag
import networkx as nx
import numpy as np
from scipy.stats import norm
import itertools


def gen_x0(prevalence_lst, rng, B, G, N = 10000, d = 10):

    """===================step 1: get the indegree dict==================="""
    topo_lst = list(nx.topological_sort(G)) # list of topological order

    # dict of the indegree of the nodes;
    # k: node number; v: in degree
    indegree_dic = {}
    for i in sorted(G.in_degree, key=lambda x: x[1]):
        indegree_dic[i[0]] = i[1]

    """===================step 2: Generate weighted adjancy matrix at t0==================="""
    x0 = np.zeros((N,d)) # data sample at t0
    w0 = np.zeros((d,d)) # weight matrix at t0
    l0 = np.zeros((d,d)) # l_{j, i} is the length of the longest path from node j to node i.
    var_dic = {i:1.0 for i in range(d)} # variance of each variable

    for node in topo_lst:
        indegree = indegree_dic[node]
        if indegree == 0:
            # nodes without any parent are sampled from normal dist
            x0[:,node] = rng.normal(loc = 0.0, scale = var_dic[node], size = N)
        else:
            # set the variance of the noise to the half of the variable
            var_noise = 0.5 * var_dic[node]
            x0_noise = rng.normal(loc=0.0, scale=np.sqrt(var_noise), size=N)

            # get the parents list of the current node
            parent_lst = list(G.predecessors(node))
            assert len(parent_lst) == indegree
            # compute the pairwise longest path
            for parent in parent_lst:
                l0[parent][node] = max([len(i) for i in nx.all_simple_paths(G,parent,node, cutoff=4)]) - 1

            # solve c - Eq (15)
            var_term_weight = np.sum([1/l0[parent][node]**2 for parent in parent_lst])
            cov_term_weight = np.sum([2/(l0[i[0]][node] * l0[i[1]][node]) * np.cov(x0[:, [i[0],i[1]]],rowvar=False)[1][0] for i in itertools.combinations(parent_lst, 2)])
            c = np.sqrt((var_dic[node] - var_noise)/(var_term_weight + cov_term_weight))

            # set weight - Eq (14)
            for parent in parent_lst:
                w0[parent][node] = c/l0[parent][node]

            x0[:, node] = x0_noise
            np.cov(x0[:, parent_lst],rowvar=False)

            for parent in parent_lst:
                x0[:, node] = x0[:, node] + w0[parent][node]*x0[:, parent]

    # check the skeleton of weighted adjacency matrix is the same as adjacency matrix
    assert np.sum(np.where(w0>0, 1, 0) - B) == 0
    # check the variance is not far from ideal value
    assert np.max(np.abs(np.var(x0,axis = 0) - np.array(list(var_dic.values())))) < 0.1

    """===================step 4: Generate mu vector at t0==================="""
    # alternative of Eq (18)
    mu = np.zeros(d)
    for node in topo_lst:
        mu[node] = -norm.ppf(1 - prevalence_lst[node])
        x0[:, node] = x0[:, node] + mu[node]

    return x0, w0, mu
