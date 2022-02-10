import numpy as np
import networkx as nx
import itertools


def gen_w(d, rng, G, x0, incidence_lst, time_steps):
    N = x0.shape[0]
    w = np.zeros((d, d))  # weight matrix at t0
    l0 = np.zeros((d,d)) # l_{j, i} is the length of the longest path from node jto node i.
    mu = np.zeros(d) # value controls the incidence rate
    gamma = np.array([rng.uniform(low=0.3, high=0.5) for i in range(d)]) # how much of the variance is explained by noise

    var_dic = {i:1.0 for i in range(d)} # variance of each variable
    xt = np.zeros_like(x0) # data sample at t0

    # set the weight of autoregression
    for i in range(d):
        # w[i][i] = rng.uniform(low=0.5, high=np.sqrt(1-gamma[i]))
        w[i][i] = rng.uniform(low=0.2, high=0.4)

    topo_lst = list(nx.topological_sort(G)) # list of topological order

    indegree_dic = {}  # list of the indegree of the nodes
    for i in sorted(G.in_degree, key=lambda x: x[1]):
        indegree_dic[i[0]] = i[1]


    for node in topo_lst:
        if indegree_dic[node] == 0:
            # For nodes without parent
            var_noise = var_dic[node] - w[node][node]**2
            xt[:, node] = xt[:, node] + w[node][node] * x0[:, node]
            xt[:, node] = xt[:, node] + rng.normal(loc=0.0, scale=np.sqrt(var_noise), size=N)
            mu[node] = incidence_lst[node] - (w[node][node] - 1) * np.mean(x0[:, node]) # eq (25)
            xt[:, node] = xt[:, node] + mu[node]
            continue

        # compute noise
        var_noise = gamma[node] * var_dic[node]
        xt[:, node] = rng.normal(loc=0.0, scale=np.sqrt(var_noise), size=N)

        # ================= solve eq (22) =================
        c = w[node][node]**2 - 1 + gamma[node]
        parent_lst =  list(G.predecessors(node))
        for parent in parent_lst:
            l0[parent][node] = max([len(i) for i in nx.all_simple_paths(G, parent, node, cutoff=4)]) - 1
        a_var_term = np.sum([1 / l0[parent][node] ** 2 for parent in parent_lst])
        a_cov_term = np.sum(
            [2 / (l0[i[0]][node] * l0[i[1]][node]) * np.cov(x0[:, [i[0], i[1]]], rowvar=False)[1][0] for i in
             itertools.combinations(parent_lst, 2)])
        a = a_var_term + a_cov_term
        b = np.sum([(w[node][node] / l0[parent][node]) *  np.cov(x0[:, [parent, node]], rowvar=False)[1][0] for parent in parent_lst])

        solution = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        # ================================================

        # set value
        for parent in parent_lst:
            w[parent][node] = solution / l0[parent][node]
        xt[:, node] = xt[:, node] + w[node][node] * x0[:, node]

        # eq (25)
        mu[node] = incidence_lst[node] - (w[node][node]-1) * np.mean(x0[:, node]) - np.sum([(w[parent][node]) * np.mean(x0[:, parent]) for parent in parent_lst])
        for parent in parent_lst:
            xt[:, node] = xt[:, node] + w[parent][node] * x0[:, parent]
        xt[:, node] = xt[:, node] + mu[node]

    # generate follow-up
    t_lst = [x0,xt]
    for i in range(time_steps-2):
        x0 = xt
        xt = np.zeros_like(x0) # data sample at t0

        for node in topo_lst:

            if indegree_dic[node] == 0:
                var_noise = var_dic[node]-w[node][node]**2
                xt[:, node] = xt[:, node] + w[node][node] * x0[:, node]
                xt[:, node] = xt[:, node] + rng.normal(loc=0.0, scale=np.sqrt(var_noise), size=N)
                xt[:, node] = xt[:, node] + mu[node]
                continue
            parent_lst = list(G.predecessors(node))
            var_noise =  gamma[node] * var_dic[node]
            xt[:, node] = rng.normal(loc=0.0, scale=np.sqrt(var_noise), size=N)

            xt[:, node] = xt[:, node] + w[node][node] * x0[:, node]
            for parent in parent_lst:
                xt[:, node] = xt[:, node] + w[parent][node] * x0[:, parent]
            xt[:, node] = xt[:, node] + mu[node]
        t_lst.append(xt)


    t_lst = np.array(t_lst)
    t_lst = t_lst.transpose(1,0,2)
    return t_lst, w, mu, gamma

