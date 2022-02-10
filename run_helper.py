from models.golem import run_golem
from models.tetrad_methods import run_tetrad
from models.notears.notear_mlp import run_notears_mlp
from models.L_lingam import run_Llingam
import numpy as np
import torch
from utils import  random_scale, create_weight, random_flip
from sklearn.utils import resample

# model_type: ["golem", "longlingam", "notears", "pc", "fges"]
def run_exp(args, data_load_path, model_type,
            golem_lambda_l1 = 1e-3, golem_lambda_h = 10, golem_lambda_p = 1e-3, golem_lr = 1e-3,
            golem_num_iter = 1000, golem_gamma = 10,
            notears_gamma = 10, notears_p1 = 1, notears_p2 = 1,
            notears_lambda1 = 1e-6, notears_lambda2 = 1e-6,
            all_bootstrap_seed = 0, all_using_weight = False, all_corrupt_level = 0.0
            ):
    precedent_matrix_ori = torch.from_numpy(
        np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()

    x1_ori = np.loadtxt("{}/x1.csv".format(data_load_path), delimiter=',').astype(np.float32)

    x1_imput_ori = np.loadtxt("{}/x1_imput.csv".format(data_load_path), delimiter=',').astype(np.float32)
    x2_imput_ori = np.loadtxt("{}/x2_imput.csv".format(data_load_path), delimiter=',').astype(np.float32)

    """======= generate input data by randomly scale the original data ===================="""
    x1_imput_scale = random_scale(x1_imput_ori, args.rng).astype(np.float32)
    x2_imput_scale  = random_scale(x2_imput_ori, args.rng).astype(np.float32)

    x1_imput, x2_imput, x1 = resample(x1_imput_scale, x2_imput_scale, x1_ori, random_state=all_bootstrap_seed)
    """======= generate precedent matrix by random flip the original data ===================="""
    precedent_matrix = random_flip(precedent_matrix_ori, all_corrupt_level, args.rng)

    if all_using_weight is False:
        weight_mat = None
    else:
        weight_mat = torch.from_numpy(create_weight(x1)).float()

    if model_type == "golem":
        """=========================== golem model ==========================="""
        w_golem_hard, log_golem_hard, model_golem_hard = \
            run_golem(x1_imput, x2_imput, precedent_matrix, golem_lambda_l1, golem_lambda_h, golem_lambda_p,
                      equal_variances=False, num_iter=golem_num_iter, learning_rate=golem_lr, dagness="yu", p_term="hard",
                      weight_matrix=weight_mat, gamma = golem_gamma)

        w_golem_soft, log_golem_soft, model_golem_soft = \
            run_golem(x1_imput, x2_imput, precedent_matrix, golem_lambda_l1, golem_lambda_h, golem_lambda_p,
                      equal_variances=False, num_iter=golem_num_iter, learning_rate=golem_lr, dagness="yu", p_term="soft",
                      weight_matrix=weight_mat, gamma = golem_gamma)

        w_golem_vanilla, log_golem_vanilla, model_golem_vanilla = \
            run_golem(x1_imput, x2_imput, precedent_matrix, golem_lambda_l1, golem_lambda_h, golem_lambda_p,
                      equal_variances=False, num_iter=golem_num_iter, learning_rate=golem_lr, dagness="yu", p_term="None",
                      weight_matrix=weight_mat, gamma = golem_gamma)
        result_dic = {"golem hard": w_golem_hard.detach().cpu().numpy(),
                      "golem soft": w_golem_soft.detach().cpu().numpy(),
                      "golem vanilla": w_golem_vanilla.detach().cpu().numpy()}
        log_dic = {"golem hard": log_golem_hard,
                   "golem soft": log_golem_soft,
                   "golem vanilla": log_golem_vanilla}

    elif model_type == "longlingam":
        w_lingam, log_lingam = run_Llingam(x1_imput, x2_imput)
        result_dic = {"longitudinal lingam":  w_lingam}
        log_dic = {"longitudinal lingam": log_lingam}

    elif model_type == "notears":
        w_notears_hard, log_notears_hard = run_notears_mlp(torch.from_numpy(x1_imput).float(),
                                                           torch.from_numpy(x2_imput).float(), precedent_matrix,
                                                           weight_matrix=weight_mat,
                                                           loss_type="mse", p_term="hard", gamma=notears_gamma,
                                                           p1=notears_p1, p2=notears_p2,
                                                           lambda1=notears_lambda1, lambda2=notears_lambda2)
        w_notears_soft, log_notears_soft = run_notears_mlp(torch.from_numpy(x1_imput).float(),
                                                           torch.from_numpy(x2_imput).float(), precedent_matrix,
                                                           weight_matrix=weight_mat,
                                                           loss_type="mse", p_term="soft",
                                                           gamma=notears_gamma,
                                                           p1=notears_p1, p2=notears_p2,
                                                           lambda1=notears_lambda1, lambda2=notears_lambda2)
        w_notears_vanilla, log_notears_vanilla = run_notears_mlp(torch.from_numpy(x1_imput).float(),
                                                                 torch.from_numpy(x2_imput).float(), precedent_matrix,
                                                                 weight_matrix=weight_mat,
                                                                 loss_type="mse", p_term="None",
                                                                 gamma=notears_gamma, p1=notears_p1, p2=notears_p2,
                                                                 lambda1=notears_lambda1, lambda2=notears_lambda2)

        result_dic = {"notears hard": w_notears_hard,
                      "notears soft": w_notears_soft,
                      "notears vanilla": w_notears_vanilla}
        log_dic = {"notears hard": log_notears_hard,
                   "notears soft": log_notears_soft,
                   "notears vanilla": log_notears_vanilla}

    elif model_type == "pc":

        w_pc_forbid, edge_lst_pc_forbid, require_lst_pc, forbid_lst_pc = \
            run_tetrad(x1_imput, x2_imput, precedent_matrix, prior_type="forbid", verbose=False,
                       method="pc", dataType='continuous')
        w_pc_none, edge_lst_pc_none, _, _ = run_tetrad(x1_imput, x2_imput, precedent_matrix,
                                                       prior_type="none", verbose=False,
                                                       method="pc", dataType='continuous')


        result_dic = {"pc forbid": w_pc_forbid,
                      "pc none": w_pc_none
                      }
        log_dic = {"pc forbid": edge_lst_pc_forbid,
                   "pc none": edge_lst_pc_none,
                   "pc forbid edge lst": forbid_lst_pc}

    elif model_type == "fges":
        w_fges_forbid, edge_lst_fges_forbid, require_lst_fges, forbid_lst_fges = \
            run_tetrad(x1_imput, x2_imput, precedent_matrix, prior_type="forbid", verbose=False,
                       method="fges", dataType='continuous')
        w_fges_none, edge_lst_fges_none, _, _ = run_tetrad(x1_imput, x2_imput, precedent_matrix,
                                                           prior_type="none", verbose=False,
                                                           method="fges", dataType='continuous')

        result_dic = {"fges forbid": w_fges_forbid,
                      "fges none": w_fges_none
                      }
        log_dic = {"fges forbid": edge_lst_fges_forbid,
                   "fges none": edge_lst_fges_none,
                   "fges forbid edge lst": forbid_lst_fges,
                   }
    else:
        raise NotImplementedError
    return result_dic, log_dic
