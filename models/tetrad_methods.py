import pandas as pd
import torch
import numpy as np
from pycausal import prior as p
from pycausal import search as s
from pycausal.pycausal import pycausal as pca


def create_forbid_and_require_list(precedent_matrix):
    forbid_lst, require_lst = [], []
    l = precedent_matrix.size(0)
    for i in range(l):
        for j in range(l):
            if precedent_matrix[i, j] == 1:
                require_lst.append(["x_{}_t1".format(i), "x_{}_t2".format(j)])
            elif precedent_matrix[i, j] == 0:
                forbid_lst.append(["x_{}_t1".format(i), "x_{}_t2".format(j)])
                forbid_lst.append(["x_{}_t1".format(i), "x_{}_t1".format(j)])
                forbid_lst.append(["x_{}_t2".format(i), "x_{}_t2".format(j)])
            else:
                print("There is none binary value {} in the precedent matrix".format(precedent_matrix[i, j]))
                raise NotImplementedError
    assert torch.sum(precedent_matrix).item() == len(require_lst)
    assert len(require_lst) + len(forbid_lst) // 3 == l ** 2
    return require_lst, forbid_lst
def random_convert_edge_lst_to_matrix(edge_lst, d):
    est_w = np.zeros((d,d))
    for i in edge_lst:
        str_tmp = i.split(" ")
        direction = str_tmp[1]
        if direction == '-->' or direction == 'o->':
            idx_out = int(str_tmp[0].split("_")[1])
            idx_in = int(str_tmp[-1].split("_")[1])
            est_w[idx_out][idx_in] = 1
        elif direction == '<--':
            idx_in = int(str_tmp[0].split("_")[1])
            idx_out = int(str_tmp[-1].split("_")[1])
            est_w[idx_out][idx_in] = 1
        elif direction == '---' or direction == '<->':
            idx_in = int(str_tmp[0].split("_")[1])
            idx_out = int(str_tmp[-1].split("_")[1])

            choice = np.random.randint(0,2)
            if choice == 0:
                est_w[idx_out][idx_in] = 1
            else:
                est_w[idx_in][idx_out] = 1

        else:
            raise NotImplementedError

    return est_w

def convert_edge_lst_to_matrix(edge_lst, precedent_matrix):
    est_w = np.zeros_like(precedent_matrix)
    for i in edge_lst:
        str_tmp = i.split(" ")
        direction = str_tmp[1]
        if direction == '-->' or direction == 'o->':
            idx_out = int(str_tmp[0].split("_")[1])
            idx_in = int(str_tmp[-1].split("_")[1])
            est_w[idx_out][idx_in] = 1
        elif direction == '<--':
            idx_in = int(str_tmp[0].split("_")[1])
            idx_out = int(str_tmp[-1].split("_")[1])
            est_w[idx_out][idx_in] = 1
        elif direction == '---' or direction == '<->':

            idx_in = int(str_tmp[0].split("_")[1])
            idx_out = int(str_tmp[-1].split("_")[1])
            if precedent_matrix[idx_in][idx_out] == 0 and precedent_matrix[idx_out][idx_in] == 1:
                est_w[idx_out][idx_in] = 1
            elif precedent_matrix[idx_in][idx_out] == 1 and precedent_matrix[idx_out][idx_in] == 0:
                est_w[idx_in][idx_out] = 1
            else:
                continue
        else:
            raise NotImplementedError

    return est_w


def create_temporal(df):
    tier1 = [i for i in list(df.columns) if i[-2:] == "t1"]
    tier2 = [i for i in list(df.columns) if i[-2:] == "t2"]
    assert len(tier1) + len(tier2) == len(df.columns)
    temporal = [tier1, tier2]
    return temporal


def run_tetrad(x1_imput, x2_imput, precedent_matrix, prior_type="forbid", verbose=False, method="fges",
             dataType='continuous'):
    assert dataType in ['discrete', 'mixed', 'continuous']
    assert method in ['fges', 'pc', "rfci"]
    print("running {} w/ {} constraint".format(method, prior_type))
    # convert numpy array into data frame
    if verbose:
        print("prepare prior knowledge ...")
    df_x1 = pd.DataFrame(x1_imput, columns=["x_{}_t1".format(i) for i in range(x1_imput.shape[1])])
    df_x2 = pd.DataFrame(x2_imput, columns=["x_{}_t2".format(i) for i in range(x1_imput.shape[1])])

    # concatenate two cross sections
    df = pd.concat([df_x1, df_x2], axis=1)

    # create prior knowledge for csd methods
    require_lst, forbid_lst = create_forbid_and_require_list(precedent_matrix)
    temporal = create_temporal(df)

    # prior = p.knowledge(forbiddirect=forbid_lst, requiredirect=require_lst, addtemporal=temporal)
    if prior_type == "forbid":
        prior = p.knowledge(forbiddirect=forbid_lst, addtemporal=temporal)
    elif prior_type == "require":
        prior = p.knowledge(requiredirect=require_lst, addtemporal=temporal)
    elif prior_type == "none":
        prior = None
    else:
        print("Please choose prior_type from [\"forbid\", \"require\"]")
        raise NotImplementedError

    if verbose:
        print("run method: {} ...".format(method))

    tetrad = s.tetradrunner()
    if method == 'fges':
        if dataType == 'discrete':
            tetrad.run(algoId='fges', dfs=df, priorKnowledge=prior, dataType=dataType,
                       maxDegree=-1, faithfulnessAssumed=True, verbose=False, scoreId='bdeu-score')
        elif dataType == 'continuous':
            tetrad.run(algoId='fges', dfs=df, priorKnowledge=prior, dataType=dataType,
                       maxDegree=-1, faithfulnessAssumed=True, verbose=False, scoreId='sem-bic')
        else:
            raise NotImplementedError
    elif method == 'rfci':
        if dataType == 'discrete':
            tetrad.run(algoId='rfci', dfs=df, priorKnowledge=prior, dataType=dataType, testId='chi-square-test',
                       depth=-1, completeRuleSetUsed = True, verbose=False)
        elif dataType == 'continuous':
            tetrad.run(algoId='pc-all', dfs=df, priorKnowledge=prior, dataType=dataType, testId='fisher-z-test',
                       depth=-1, completeRuleSetUsed = True, verbose=False)
        else:
            raise NotImplementedError
    elif method == 'pc':
        if dataType == 'discrete':

            tetrad.run(algoId='pc-all', dfs=df, priorKnowledge=prior, dataType=dataType, testId='chi-square-test',
                       depth=2, concurrentFAS=True,
                       useMaxPOrientationHeuristic=True, verbose=False)
        elif dataType == 'continuous':
            tetrad.run(algoId='pc-all', dfs=df, priorKnowledge=prior, dataType=dataType, testId='fisher-z-test',
                       depth=2, concurrentFAS=True,
                       useMaxPOrientationHeuristic=True, verbose=False)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    edge_lst = tetrad.getEdges()

    est_w = convert_edge_lst_to_matrix(edge_lst, precedent_matrix)
    if verbose:
        print("DONE!")
    return est_w, edge_lst, require_lst, forbid_lst


if __name__ == "__main__":
    seed_graph, seed = 123, 123
    followup_min = 6
    d = 16
    N = 1000
    followup_max, delta_t = 24, 12

    data_load_path = "/home/kumarbio/yang6993/exp/csd/syn_data/{}_{}_{}_{}_{}_{}_{}/".format(
        seed_graph, seed, N, d, followup_min, followup_max, delta_t)
    B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
    precedent_matrix = torch.from_numpy(
        np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()

    x = np.loadtxt("{}/x_agg.csv".format(data_load_path), delimiter=',').astype(np.float32)
    x1 = np.loadtxt("{}/x1.csv".format(data_load_path), delimiter=',').astype(np.float32)
    x2 = np.loadtxt("{}/x2.csv".format(data_load_path), delimiter=',').astype(np.float32)

    x_imput = np.loadtxt("{}/x_agg_imput.csv".format(data_load_path), delimiter=',').astype(np.float32)
    x1_imput = np.loadtxt("{}/x1_imput.csv".format(data_load_path), delimiter=',').astype(np.float32)
    x2_imput = np.loadtxt("{}/x2_imput.csv".format(data_load_path), delimiter=',').astype(np.float32)

    pc = pca()
    pc.start_vm()
    est_w_fges, edge_lst_fges, require_lst_fges, forbid_lst_fges = run_tetrad(x1_imput, x2_imput, precedent_matrix, prior_type="forbid", verbose=True, method="fges",
                          dataType='continuous')

    est_w_pc, edge_lst_pc, require_lst_fges_pc, forbid_lst_fges_pc = run_tetrad(x1_imput, x2_imput, precedent_matrix, prior_type="forbid", verbose=True, method="pc",
                        dataType='continuous')

    pc.stop_vm()
