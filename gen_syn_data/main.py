from gen_dag import gen_dag
from gen_x0 import gen_x0
from obs_mask import gen_obs_mask
from gen_w import gen_w
from gen_precedence_matrix import gen_p, get_x_agg
import numpy as np
import networkx as nx
import os


# for d in [8,16,32]:
#     for N in [1000, 5000]:

# time_steps = 100
# d = 8  # number of features
# N = 1000  # number of samples
# followup_min = 6 # min length of the followup window
# followup_max = 24 # max length of the followup window
# t_init = 40
# delta_t = 12  # the length of the cross-section, should be smaller than followup_max
#
# # the significant threshold to add an edge into precedence matrix, control the density of the precedence matrix
# significant_threshold = 0.01
#
# # the threshold of the number of feature count, also control the density of the precedence matrix
# max_threshold = 0.1 * N
# # random seed to generate the graph skeleton
# seed_graph = 123
# # random seed to generate the scm
# seed_scm = 123
#
# # the density of the edge in the summary graph
# edge_density = 0.5
# save_dir = "/Users/yanghaoyu/Library/Mobile Documents/com~apple~CloudDocs/code/csd/syn_data"
# save_folder = "/{}_{}_{}_{}_{}_{}_{}/".format(seed_graph, seed_scm, N, d,followup_min,followup_max,delta_t)

def gen_syn_data(time_steps: "number of time step in time series" = 100,
                 d: "number of features" = 8, N: "number of samples" = 1000,
                 followup_min: "min length of the followup window" = 6,
                 followup_max: "max length of the followup window" = 24,
                 t_init: "the starting point to choose the followup window" = 40,
                 delta_t: "the length of the cross-section, should be smaller than followup_max" = 12,
                 significant_threshold: "the significant threshold to add an edge into precedence matrix, "
                                        "control the density of the precedence matrix" = 0.01,
                 seed_graph: "random seed to generate the graph skeleton" = 123,
                 seed_scm: "random seed to generate the scm" = 1,
                 edge_density: "the density of the edge in the summary graph" = 0.5,
                 save_dir: "dir to save generated data" = "/Users/yanghaoyu/Library/Mobile Documents/com~apple~CloudDocs/code/csd/syn_data",
                 save_folder: "folder to save generated data" = "",
                 verbose = False
                 ):
    # the threshold of the number of feature count, also control the density of the precedence matrix
    max_threshold = 0.1 * N

    save_path = save_dir + save_folder
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    rng_graph = np.random.default_rng(seed_graph)

    rng_scm = np.random.default_rng(seed_scm)

    """===================step 1: Generate DAG==================="""
    if verbose:
        print("Generate DAG ... ")
    # generate DAG
    B, G = gen_dag(rng_graph, seed_graph, d, edge_density)

    """===================step 2: Generate Prevalence List==================="""
    if verbose:
        print("Generate Prevalence List ... ")
    topo_lst = list(nx.topological_sort(G))

    # prevalence_lst is set according to topological order
    prevalence_lst = np.zeros(len(topo_lst))

    for idx, node in enumerate(topo_lst):
        prevalence_lst[node] = np.exp(-idx / d) - 0.3

    """===================step 3: Generate initial step==================="""
    if verbose:
        print("Generate initial step ... ")
    x0, w0, mu0 = gen_x0(prevalence_lst, rng_scm, B, G, N=N, d=d)

    """===================step 4: Generate following steps==================="""
    if verbose:
        print("Generate following steps ... ")
    # incidence rate list is set according to prevalence at t0
    incidence_lst = prevalence_lst / 10
    x, w, mu, gamma = gen_w(d, rng_scm, G, x0, incidence_lst, time_steps)

    """===================step 5: Generate observation mask==================="""
    if verbose:
        print("Generate observation mask ... ")
    observation_mask, log_lst = gen_obs_mask(rng_scm, x, topo_lst, d, N, time_steps,
                                             followup_min=followup_min,
                                             followup_max=followup_max)

    x_mask = np.multiply(observation_mask, x)

    """===================step 6: Generate precedent matrix==================="""
    if verbose:
        print("Generate precedent matrix ... ")
    precede_matrix, precede_matrix_soft, df_t1, df_t2 = gen_p(x_mask, d, t_init=t_init, delta_t=delta_t,
                                                              significant_threshold=significant_threshold,
                                                              max_threshold=max_threshold)
    x1, x2, x_mean = get_x_agg(df_t1.copy(), df_t2.copy())


    """===================step 7:imputation==================="""
    if verbose:
        print("Imputation ... ")

    x1_imput, x2_imput = x1.copy(), x2.copy()
    # fill na with mean
    for i in range(d):
        x1_imput[:, i][np.isnan(x1_imput[:, i])] = x_mean[i]
        x2_imput[:, i][np.isnan(x2_imput[:, i])] = x_mean[i]


    np.savetxt(save_path + "B.csv", B, delimiter=",")
    np.savetxt(save_path + "w.csv", w, delimiter=",")
    np.savetxt(save_path + "x1.csv", x1, delimiter=",")
    np.savetxt(save_path + "x2.csv", x2, delimiter=",")
    np.savetxt(save_path + "x1_imput.csv", x1_imput, delimiter=",")
    np.savetxt(save_path + "x2_imput.csv", x2_imput, delimiter=",")
    np.savetxt(save_path + "x_mean.csv", x_mean, delimiter=",")
    np.savetxt(save_path + "precede_matrix.csv", precede_matrix, delimiter=",")
    np.savetxt(save_path + "precede_matrix_soft.csv", precede_matrix_soft, delimiter=",")


"""=============================END============================="""

if __name__ == "__main__":
    cwd = os.getcwd()
    save_dir = cwd + "/syn_data"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    followup_min = 4
    followup_max = 24
    delta_t = 18
    t_init = 30
    time_steps = 100
    significant_threshold = 0.001

    for seed_graph in range(1, 6):
        for d in [16]:
            for N in [1000]:
                save_folder = "/{}_{}_{}_{}_{}_{}_{}/".format(
                    seed_graph, N, d, followup_min , followup_max, delta_t, significant_threshold)
                print("generate synthetic data with setting {}".format(save_folder))
                gen_syn_data(d=d, N=N, seed_graph=seed_graph,
                             edge_density=0.8, save_dir=save_dir, save_folder=save_folder,
                             significant_threshold=significant_threshold,
                             followup_min=followup_min, followup_max=followup_max, delta_t=delta_t, t_init=t_init,
                             time_steps=time_steps,
                             verbose=True)
