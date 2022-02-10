import numpy as np
import torch
import os
import pickle

class exp_settings:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.seed_scm = 1
        self.seed_graph_upper = 6




        self.followup_min = 4
        self.followup_max = 24
        self.delta_t = 18
        self.t_init = 30
        self.time_steps = 100
        self.significant_threshold_lst = (0.001, 0.1)

        "=============== setting for golem ==============="
        self.golem_lr_lst = (1e-3, 1e-4, 1e-5, 1e-6)
        self.golem_lambda_l1 = 1e-3
        self.golem_lambda_h = 10
        self.golem_lambda_p = 1e-3
        self.golem_num_iter = 1000


        "=============== setting for gamma ==============="
        self.notears_lamb_lst = (1e-1, 1e-3, 1e-5)
        self.notears_p1 = 1
        self.notears_p2 = 1

        "=============== setting for all ==============="
        self.all_bootstrap_seed_lst = tuple(range(10))
        self.all_using_weight_lst = (False,)
        self.N_lst = (1000,)
        self.all_corrupt_level_lst = (0.0, 0.05, 0.15, 0.25, 0.35)
        self.gamma_lst = (10,)


def create_weight(x):
    return 1 - np.isnan(x)


def random_scale(x, rng, type = "uniform"):
    d = x.shape[1]
    if type == "uniform":
        scales = rng.uniform(low=1.0, high=100.0, size=d)
    else:
        raise NotImplementedError
    x = x*scales
    # x = (x - np.mean(x,axis=0))/np.std(x,axis=0)
    return x

def random_flip(precedent_matrix, p, rng):
    d = precedent_matrix.shape[0]
    mask = rng.binomial(1, p, d * d).reshape(d, d)
    prec_mat_shuffled = precedent_matrix.detach().cpu().numpy().copy()

    for i in range(d):
        for j in range(d):
            if mask[i][j] == 1:
                prec_mat_shuffled[i][j] = 1- prec_mat_shuffled[i][j]
                # prec_mat_shuffled[j][i] = 1 - prec_mat_shuffled[j][i]
                # mask[j][i] = 0
    return torch.from_numpy(prec_mat_shuffled).float()

def save_result(cwd, model_type, log_dic, result_dic, model_name):
    """ =========== save result ============"""
    result_save_path = cwd + "/result"
    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)

    result_save_model_path = result_save_path + "/{}".format(model_type)
    if not os.path.isdir(result_save_model_path):
        os.makedirs(result_save_model_path)
    with open(result_save_model_path + "/log_{}.pkl".format(model_name), "wb") as f:
        pickle.dump(log_dic, f)

    with open(result_save_model_path + "/result_{}.pkl".format(model_name), "wb") as f:
        pickle.dump(result_dic, f)

def save_result_real(cwd, model_type, log_dic, result_dic, model_name):
    """ =========== save result ============"""
    result_save_path = cwd + "/result_real"
    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)

    result_save_model_path = result_save_path + "/{}".format(model_type)
    if not os.path.isdir(result_save_model_path):
        os.makedirs(result_save_model_path)
    with open(result_save_model_path + "/log_{}.pkl".format(model_name), "wb") as f:
        pickle.dump(log_dic, f)

    with open(result_save_model_path + "/result_{}.pkl".format(model_name), "wb") as f:
        pickle.dump(result_dic, f)
def load_result(cwd, model_type, model_name):
    """ =========== save result ============"""
    result_save_path = cwd + "/result"
    if not os.path.isdir(result_save_path):
        os.makedirs(result_save_path)

    result_save_model_path = result_save_path + "/{}".format(model_type)

    with open(result_save_model_path + "/log_{}.pkl".format(model_name), "rb") as f:
        log_dic = pickle.load(f)

    with open(result_save_model_path + "/result_{}.pkl".format(model_name), "rb") as f:
        result_dic = pickle.load(f)
    return result_dic, log_dic


