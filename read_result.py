import os
import pickle
from plot import count_accuracy, return_2d_idx, return_top_n_idx
import numpy as np
import torch
from utils import exp_settings, random_scale, create_weight, random_flip, load_result
import matplotlib.pyplot as plt
from models.tetrad_methods import random_convert_edge_lst_to_matrix
import numpy as np
import scipy.stats as st


def binarize_w(w, threshold=0.1):
    w[w < threshold] = 0.0
    w[w >= threshold] = 1.0
    return w


def threshold_score(score_dic_threshold, B, model_type, model_name, threshold=0.1):
    cwd = os.getcwd()

    with open(cwd + "/result/{}/result_{}.pkl".format(model_type, model_name), "rb") as f:
        result_dic_tmp = pickle.load(f)

    for k, v in result_dic_tmp.items():
        result_tmp = count_accuracy(B, binarize_w(v.copy(), threshold=threshold))
        for k2, v2 in result_tmp.items():
            score_dic_threshold.setdefault(k2, {})
            score_dic_threshold[k2][k] = v2
        score_dic_threshold.setdefault("prec", {})
        score_dic_threshold["prec"][k] = 1 - score_dic_threshold["fdr"][k]
    return score_dic_threshold


def threshold_score_pc(score_dic_threshold, B, model_type, model_name, d, threshold=0.1):
    cwd = os.getcwd()

    # with open(cwd + "/result/{}/result_{}.pkl".format(model_type, model_name), "rb") as f:
    #     result_dic_tmp = pickle.load(f)
    result_dic_pc, log_dic_pc = load_result(cwd, model_type, model_name)
    for k, v in log_dic_pc.items():
        if "edge" in k:
            continue
        result_tmp = count_accuracy(B, random_convert_edge_lst_to_matrix(v, d))
        for k2, v2 in result_tmp.items():
            score_dic_threshold.setdefault(k2, {})
            score_dic_threshold[k2][k] = v2
        score_dic_threshold.setdefault("prec", {})
        score_dic_threshold["prec"][k] = 1 - score_dic_threshold["fdr"][k]
    return score_dic_threshold


def n_edges_score(score_dic_nedges, n, B, model_type, model_name):
    cwd = os.getcwd()

    with open(cwd + "/result/{}/result_{}.pkl".format(model_type, model_name), "rb") as f:
        result_dic_tmp = pickle.load(f)

    for k, v in result_dic_tmp.items():
        "=========== method using number of edge ==================="
        w_nedge = v.copy()
        top_n_idx = return_top_n_idx(w_nedge, n)
        for i in range(len(w_nedge)):
            for j in range(len(w_nedge)):
                if (i, j) in top_n_idx:
                    w_nedge[i][j] = 1
                else:
                    w_nedge[i][j] = 0
        assert np.sum(w_nedge) == n
        result_tmp = count_accuracy(B, w_nedge)
        for k2, v2 in result_tmp.items():
            score_dic_nedges.setdefault(k2, {})
            score_dic_nedges[k2][k] = v2
        score_dic_nedges["prec"][k] = 1 - score_dic_nedges["fdr"][k]
    return score_dic_nedges


def curve_score(curve_dic, B, model_type, model_name, model_save_name):
    cwd = os.getcwd()
    with open(cwd + "/result/{}/result_{}.pkl".format(model_type, model_save_name), "rb") as f:
        result_dic_tmp = pickle.load(f)
    v = result_dic_tmp[model_name]

    """============== for plotting curve ==========="""
    w_curve = v.copy()
    top_n = len(B) ** 2
    top_n_idx = return_top_n_idx(w_curve, top_n)
    curve_dic.setdefault(model_name, {})
    # curve_lst = []
    for i in range(top_n):
        B_est = np.zeros_like(w_curve)
        for j in range(i + 1):
            B_est[top_n_idx[j][0], top_n_idx[j][1]] = 1
        result_tmp = count_accuracy(B, B_est)

        # for k, v in result_tmp.items():
        for k2, v2 in result_tmp.items():
            curve_dic[model_name].setdefault(k2, []).append(v2)
        curve_dic[model_name].setdefault("prec", []).append(1 - result_tmp["fdr"])

        # curve_lst.append(result_tmp["tpr"])
    # curve_dic[model_name_tmp].setdefault("tpr", []).append(curve_lst)

    return curve_dic


def get_best_result(args, N, d, seed_graph, all_corrupt_level, all_using_weight, all_bootstrap_seed,
                    significant_threshold,
                    golem_gamma_lst=(1, 5, 10), notears_gamma_lst=(1, 5, 10)):
    cwd = os.getcwd()
    best_result = {}
    """Find best model"""
    model_type = "golem"

    for golem_lr in args.golem_lr_lst:
        for golem_gamma in golem_gamma_lst:
            model_name_golem = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                               all_corrupt_level,
                                                                               all_bootstrap_seed,
                                                                               significant_threshold,
                                                                               golem_gamma,
                                                                               args.golem_lambda_l1,
                                                                               args.golem_lambda_h,
                                                                               args.golem_lambda_p,
                                                                               golem_lr,
                                                                               args.golem_num_iter,
                                                                               all_using_weight)
            result_dic_golem, log_dic_golem = load_result(cwd, model_type, model_name_golem)
            for k, v in log_dic_golem.items():
                best_result.setdefault(k, {"metric": np.inf, "model name": None})
                metric_tmp = v['score'][-1]
                if metric_tmp <= best_result[k]["metric"]:
                    best_result[k]["metric"] = metric_tmp
                    best_result[k]["model name"] = model_name_golem

    model_type = "notears"
    for lamb in args.notears_lamb_lst:
        for notears_gamma in notears_gamma_lst:
            notears_lambda1, notears_lambda2 = lamb, lamb
            model_name_notears = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                              all_corrupt_level,
                                                                              all_bootstrap_seed,
                                                                              significant_threshold,
                                                                              notears_gamma,
                                                                              args.notears_p1,
                                                                              args.notears_p2,
                                                                              notears_lambda1,
                                                                              notears_lambda2,
                                                                              all_using_weight)
            result_dic_notears, log_dic_notears = load_result(cwd, model_type, model_name_notears)
            for k, v in log_dic_notears.items():
                best_result.setdefault(k, {"metric": np.inf, "model name": None})
                metric_tmp = v['loss'][-1][-1]
                if metric_tmp <= best_result[k]["metric"]:
                    best_result[k]["metric"] = metric_tmp
                    best_result[k]["model name"] = model_name_notears
    return best_result


def get_sort_weight_dic(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight, all_bootstrap_seed=0):
    cwd = os.getcwd()

    weight_dic = {}
    """-------------------- golem & notears --------------------"""
    for model_type in ["notears", "golem"]:
        for gamma in [1, 5, 10]:
            if model_type == "golem":
                model_name_golem = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                                all_corrupt_level,
                                                                                args.golem_lambda_l1,
                                                                                args.golem_lambda_h,
                                                                                args.golem_lambda_p,
                                                                                1e-3, args.golem_num_iter,
                                                                                gamma,
                                                                                all_using_weight,
                                                                                all_bootstrap_seed)
                result_dic, log_dic_golem = load_result(cwd, model_type, model_name_golem)


            elif model_type == "notears":
                model_name_notears = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                               all_corrupt_level, gamma,
                                                                               args.notears_p1, args.notears_p2,
                                                                               1e-3, 1e-3,
                                                                               all_using_weight, all_bootstrap_seed)
                result_dic, log_dic_notears = load_result(cwd, model_type, model_name_notears)
            else:
                continue
            for model_name_ori, weight_mat in result_dic.items():
                if "hard" in model_name_ori:
                    if gamma != 1:
                        continue

                if "vanilla" in model_name_ori:
                    if gamma != 1:
                        continue
                    model_name = model_name_ori.split(" ")[0]
                    weight_dic[model_name] = sorted(abs(weight_mat.flatten()))
                else:
                    model_name = model_name_ori
                    weight_dic[model_name + " gamma = {}".format(gamma)] = sorted(abs(weight_mat.flatten()))

    model_type = "longlingam"

    model_name_longlingam = "{}_{}_{}_{}".format(N, d, seed_graph, all_bootstrap_seed)
    with open(cwd + "/result/{}/result_{}.pkl".format(model_type, model_name_longlingam), "rb") as f:
        result_dic_tmp = pickle.load(f)
    model_name = 'longitudinal lingam'
    weight_mat = result_dic_tmp[model_name]
    weight_dic[model_name] = sorted(abs(weight_mat.flatten()))
    return weight_dic


def get_score_dic(args, precedent_matrix, B, N, d, seed_graph, all_corrupt_level, all_using_weight,
                  significant_threshold, golem_gamma_lst, notears_gamma_lst, threshold=0.05):
    """============================ print scores ================================"""
    score_dic_boot = {}

    """ =========== load data ============"""
    for all_bootstrap_seed in args.all_bootstrap_seed_lst:

        # find the best result for golem and notears
        best_result = get_best_result(args, N, d, seed_graph, all_corrupt_level, all_using_weight, all_bootstrap_seed,
                                      significant_threshold, golem_gamma_lst=golem_gamma_lst, notears_gamma_lst=notears_gamma_lst)

        # get the score given a threshold
        score_dic = {}
        """-------------------- pc --------------------"""
        model_type = "pc"
        model_name_pc = "{}_{}_{}_{}_{}_{}".format(N, d, seed_graph, all_corrupt_level, all_bootstrap_seed, significant_threshold)
        if "top" in str(threshold):
            threshold_pc = 0.1
        else:
            threshold_pc = threshold
        score_dic = threshold_score_pc(score_dic, B, model_type, model_name_pc, d, threshold=threshold_pc)

        """-------------------- fges --------------------"""
        model_type = "fges"
        model_name_fges =  "{}_{}_{}_{}_{}_{}".format(N, d, seed_graph, all_corrupt_level, all_bootstrap_seed,significant_threshold)
        if "top" in str(threshold):
            threshold_fges = 0.1
        else:
            threshold_fges = threshold
        score_dic = threshold_score(score_dic, B, model_type, model_name_fges, threshold=threshold_fges)

        """-------------------- longlingam --------------------"""
        model_type = "longlingam"
        model_name_longlingam ="{}_{}_{}_{}_{}".format(N, d, seed_graph, all_corrupt_level, all_bootstrap_seed,
                                                                    significant_threshold)
        if "top" in str(threshold):
            N_lingam = int(threshold.split(" ")[1])
            score_dic = n_edges_score(score_dic, N_lingam, B, model_type, model_name_longlingam)
        else:
            score_dic = threshold_score(score_dic, B, model_type, model_name_longlingam, threshold=threshold)

        """-------------------- golem & notears --------------------"""

        for k, v in best_result.items():
            model_type = k.split(" ")[0]
            if "top" in str(threshold):

                N_nn = int(threshold.split(" ")[1])
                score_dic = n_edges_score(score_dic, N_nn, B, model_type, v['model name'])
            else:
                score_dic = threshold_score(score_dic, B, model_type, v['model name'], threshold=threshold)

        result_tmp = count_accuracy(B, random_flip(precedent_matrix, all_corrupt_level, args.rng))
        for k2, v2 in result_tmp.items():
            score_dic.setdefault(k2, {})
            score_dic[k2]["Precedence Matrix"] = v2
        score_dic["prec"]["Precedence Matrix"] = 1 - score_dic["fdr"]["Precedence Matrix"]
        for metric_name, v in score_dic.items():
            for model_name, metric_score in v.items():
                score_dic_boot.setdefault(model_name, {})
                score_dic_boot[model_name].setdefault(metric_name, []).append(metric_score)
    return score_dic_boot


def get_curve_dic(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight,significant_threshold,golem_gamma_lst,notears_gamma_lst):
    metric_name_lst = ['fdr', 'tpr', 'trpr', 'fpr', 'shd', 'pred_size', "prec"]
    curve_dic_boot = {}
    for all_bootstrap_seed in args.all_bootstrap_seed_lst:
        best_result = get_best_result(args, N, d, seed_graph, all_corrupt_level, all_using_weight,
                                      all_bootstrap_seed,
                                      significant_threshold, golem_gamma_lst=golem_gamma_lst, notears_gamma_lst=notears_gamma_lst)
        curve_dic = {}
        """-------------------- golem & notears --------------------"""
        for model_name, v in best_result.items():
            model_save_name = v['model name']
            model_type = model_name.split(" ")[0]
            curve_dic = curve_score(curve_dic, B, model_type, model_name, model_save_name)

        model_type = "longlingam"

        model_name_longlingam = "{}_{}_{}_{}".format(N, d, seed_graph, all_bootstrap_seed)
        curve_dic = curve_score(curve_dic, B, model_type, 'longitudinal lingam', model_name_longlingam)

        for metric_name in metric_name_lst:

            for model_name, v in curve_dic.items():
                curve_dic_boot.setdefault(model_name, {})
                curve_dic_boot[model_name].setdefault(metric_name, []).append(v[metric_name])
    return curve_dic_boot


def plot_figure_add_edges(save_path, B, score_dic_boot, curve_dic_boot, new_name, ori_name, metric_name="fdr",
                          ylabel="FDR", xlabel="number of included edges", show_figure=True, fig_name="fig",
                          dotted_name_lst = ["Precedence", "fges",  "pc"]):
    plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot([result_dic["precedent matrix"]["tpr"] for i in range(top_n)], "--", label="precedent matrix")
    plt.plot([np.sum(B) for i in range(100)], [i * 0.01 for i in range(100)], "--", label="#True Edges")

    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']

    con = 0

    for model_name in score_dic_boot.keys():

        if model_name not in curve_dic_boot.keys():
            color_name = model_name.split(" ")[0]
            color_idx = dotted_name_lst.index(color_name)
            # plot dot
            y_score_mean = np.mean(score_dic_boot[model_name][metric_name])
            x_score_mean = np.mean(score_dic_boot[model_name]["pred_size"])
            try:
                if "forbid" in model_name:
                    plt.scatter(x_score_mean, y_score_mean, label=new_name[ori_name.index(model_name)],
                                color=color_lst[len(color_lst) - color_idx - 1],marker="s")
                else:
                    plt.scatter(x_score_mean, y_score_mean, label=new_name[ori_name.index(model_name)],
                                color=color_lst[len(color_lst) - color_idx - 1],marker="o")

            except:
                continue
        else:
            data_tmp = np.array(curve_dic_boot[model_name][metric_name])
            mean_tmp = np.mean(data_tmp, axis=0)
            std_tmp = np.std(data_tmp, axis=0)

            plt.plot(mean_tmp, label=new_name[ori_name.index(model_name)], color=color_lst[con])

            plt.fill_between([i for i in range(len(mean_tmp))],
                                     [max(i, 0) for i in mean_tmp - std_tmp],
                                     [min(i, 1) for i in mean_tmp + std_tmp],
                                     alpha=0.1, color=color_lst[con])
            con += 1


    plt.legend(loc=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_figure:
        plt.show()
    else:
        plt.savefig(save_path + "{}.pdf".format(fig_name))
    plt.close()

def plot_figure_add_edges_old(save_path, B, score_dic_boot, curve_dic_boot, new_name, ori_name, metric_name="fdr",
                          ylabel="FDR", xlabel="number of included edges", show_figure=True, fig_name="fig"):
    plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot([result_dic["precedent matrix"]["tpr"] for i in range(top_n)], "--", label="precedent matrix")
    plt.plot([np.sum(B) for i in range(100)], [i * 0.01 for i in range(100)], "--", label="#True Edges")

    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
    con = 0

    for model_name in score_dic_boot.keys():
        if model_name not in curve_dic_boot.keys():
            # plot dot
            y_score_mean = np.mean(score_dic_boot[model_name][metric_name])
            # y_score_std = np.std(score_dic_boot[model_name][metric_name])

            x_score_mean = np.mean(score_dic_boot[model_name]["pred_size"])
            # x_score_std = np.std(score_dic_boot[model_name]["pred_size"])
            try:
                plt.scatter(x_score_mean, y_score_mean, label=new_name[ori_name.index(model_name)])
            except:
                continue
        else:
            data_tmp = np.array(curve_dic_boot[model_name][metric_name])
            mean_tmp = np.mean(data_tmp, axis=0)
            std_tmp = np.std(data_tmp, axis=0)
            plt.plot(mean_tmp, label=new_name[ori_name.index(model_name)], color=color_lst[con])


            plt.fill_between([i for i in range(len(mean_tmp))],
                             [max(i, 0) for i in mean_tmp - std_tmp],
                             [min(i, 1) for i in mean_tmp + std_tmp],
                             alpha=0.1, color=color_lst[con])

            con += 1
    # for k, v in w_dic_point.items():
    #     plt.scatter([np.sum(v)], [result_dic[k]["tpr"]], label=k)
    plt.legend(loc=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_figure:
        plt.show()
    else:
        plt.savefig(save_path + "{}.pdf".format(fig_name))
    plt.close()


def write_table(ori_name, new_name,significant_threshold, all_corrupt_level_lst,
                golem_gamma_lst,notears_gamma_lst,
                topk_setting = None, d=16, N=1000):
    cwd = os.getcwd()
    args = exp_settings(123)


    # all_using_weight = False
    # all_corrupt_level = 0.25

    print_dic = {}
    all_using_weight = False

    print_dic.setdefault("header", "Metric")
    print_dic.setdefault("threshold", "Threshold")
    print_dic.setdefault("corruption level", "Corruption Level")
    for all_corrupt_level in all_corrupt_level_lst:
        score_dic_boot = {}
        for seed_graph in range(1, args.seed_graph_upper):
            save_folder = "/{}_{}_{}_{}_{}_{}_{}".format(seed_graph, N, d,
                                                         args.followup_min,
                                                         args.followup_max, args.delta_t,
                                                         significant_threshold)
            data_load_path = cwd + "/syn_data" + save_folder
            precedent_matrix_ori = torch.from_numpy(
                np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()
            """======= generate precedent matrix by random flip the original data ===================="""

            B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
            # score_dic_boot = get_score_dic(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight, threshold=0.05)
            if topk_setting is None:
                topk = int(np.sum(B))
            else:
                topk = topk_setting
            score_dic_tmp = get_score_dic(args, precedent_matrix_ori, B, N, d, seed_graph, all_corrupt_level,
                                          all_using_weight, significant_threshold, golem_gamma_lst,
                                          notears_gamma_lst, threshold= "top {}".format(topk))
            for model_name, v in score_dic_tmp.items():
                if "require" in model_name:
                    continue
                for metric_name, metric_val_lst in v.items():
                    score_dic_boot.setdefault(model_name, {})
                    score_dic_boot[model_name].setdefault(metric_name, [])
                    score_dic_boot[model_name][metric_name] += metric_val_lst


        all_using_weight = False

        print_dic.setdefault("header", "Metric")
        print_dic.setdefault("corruption level", "Corruption Level")
        print_dic["corruption level"] += "&"
        print_dic["corruption level"] += "\multicolumn{5}{c|}{" + str(all_corrupt_level) + "}"
        for require_metric_name in ["prec", "tpr", 'pred_size']:
            #             print(require_metric_name)

            print_dic["header"] += "&"
            if require_metric_name == 'pred_size':
                print_dic["header"] += "Prediction Size"
            elif require_metric_name == 'prec':
                print_dic["header"] += "\multicolumn{2}{c|}{Precision}"
            elif require_metric_name == 'tpr':
                print_dic["header"] += "\multicolumn{2}{c|}{Recall}"
            else:
                continue
            for model_name, v in score_dic_boot.items():

                name_idx = ori_name.index(model_name)
                print_dic.setdefault(new_name[name_idx], new_name[name_idx])

                score_lst = v[require_metric_name]

                print_dic[new_name[name_idx]] += "&"
                a, b = np.quantile(score_lst, [0.025, 0.975])
                if require_metric_name == 'pred_size':
                    print_dic[new_name[name_idx]] += "{:.2f}".format(np.mean(score_lst))
                else:

                    print_dic[new_name[name_idx]] += "{:.2f} & [{:.2f}, {:.2f}] ".format(np.mean(score_lst),
                                                                              a, b)
    for k, v in print_dic.items():
        print_dic[k] = v + "\\\\"

    print(print_dic["corruption level"])
    print(print_dic["header"])

    for k in new_name:
        print(print_dic[k])

def write_table_topK(ori_name, new_name, d=16, N=1000, corrupt_level_lst=(0.0, 0.25)):
    print_dic = {}
    all_using_weight = False

    print_dic.setdefault("header", "Metric")
    print_dic.setdefault("threshold", "Threshold")
    print_dic.setdefault("corruption level", "Corruption Level")

    cwd = os.getcwd()
    args = exp_settings(123)
    seed_scm = args.seed_scm
    print_dic = {}
    for all_corrupt_level in corrupt_level_lst:
        score_dic_boot = {}
        for seed_graph in range(1, 6):
            # all_using_weight = False
            # all_corrupt_level = 0.25
            save_folder = "/{}_{}_{}_{}_{}_{}_{}_{}".format(seed_graph, seed_scm, N, d,
                                                            args.followup_min,
                                                            args.followup_max, args.delta_t,
                                                            args.significant_threshold)
            data_load_path = cwd + "/syn_data" + save_folder
            precedent_matrix_ori = torch.from_numpy(
                np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()
            """======= generate precedent matrix by random flip the original data ===================="""

            B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
            # score_dic_boot = get_score_dic(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight, threshold=0.05)
            topk = np.sum(B)
            score_dic_tmp = get_score_dic(args, precedent_matrix_ori, B, N, d, seed_graph, all_corrupt_level,
                                          all_using_weight, threshold="top {}".format(int(topk)))

            for model_name, v in score_dic_tmp.items():
                if "require" in model_name:
                    continue
                for metric_name, metric_val_lst in v.items():
                    score_dic_boot.setdefault(model_name, {})
                    score_dic_boot[model_name].setdefault(metric_name, [])
                    score_dic_boot[model_name][metric_name] += metric_val_lst

        all_using_weight = False

        print_dic.setdefault("header", "Metric")
        print_dic.setdefault("corruption level", "Corruption Level")
        for require_metric_name in ["prec", "tpr", 'pred_size']:
            #             print(require_metric_name)
            print_dic["corruption level"] += "&"
            print_dic["corruption level"] += str(all_corrupt_level)
            print_dic["header"] += "&"
            print_dic["header"] += require_metric_name

            for model_name, v in score_dic_boot.items():
                name_idx = ori_name.index(model_name)
                print_dic.setdefault(new_name[name_idx], new_name[name_idx])

                score_lst = v[require_metric_name]

                print_dic[new_name[name_idx]] += "&"
                a, b = np.quantile(score_lst, [0.025, 0.975])

                print_dic[new_name[name_idx]] += "{:.2f} & [{:.2f}, {:.2f}] ".format(np.mean(score_lst),
                                                                                     a, b)
        for k, v in print_dic.items():
            print_dic[k] = v + "\\\\"

    """============================ plot table ================================"""

    print(print_dic["corruption level"])
    print(print_dic["header"])

    for k in new_name:
        print(print_dic[k])



def plot_graph_corrupt(args, new_name, ori_name,significant_threshold,golem_gamma_lst,notears_gamma_lst,
                       seed_graph=1, N=1000, d=16, all_using_weight=False,
                       show_figure=True, clevel=0.25,
                       metric_name="fdr", ylabel="FDR", xlabel="number of included edges"):
    cwd = os.getcwd()
    figure_save_path = cwd + "/figure/"
    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)

    save_folder = "/{}_{}_{}_{}_{}_{}_{}".format(seed_graph, N, d,
                                                    args.followup_min,
                                                    args.followup_max, args.delta_t,
                                                    significant_threshold)
    data_load_path = cwd + "/syn_data" + save_folder

    B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
    precedent_matrix_ori = torch.from_numpy(
        np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()

    score_dic_boot = get_score_dic(args, precedent_matrix_ori, B, N, d, seed_graph, clevel,
                                   all_using_weight,significant_threshold, args.gamma_lst, args.gamma_lst)

    metric_name_lst = ['tpr', 'pred_size', "prec"]
    curve_dic_boot1 = {}
    for all_bootstrap_seed in args.all_bootstrap_seed_lst:
        best_result = get_best_result(args, N, d, seed_graph, 0.0, all_using_weight,
                                      all_bootstrap_seed,
                                      significant_threshold, golem_gamma_lst=golem_gamma_lst, notears_gamma_lst=notears_gamma_lst)
        curve_dic = {}
        """-------------------- golem & notears --------------------"""
        for model_name, v in best_result.items():
            model_save_name = v['model name']
            model_type = model_name.split(" ")[0]
            curve_dic = curve_score(curve_dic, B, model_type, model_name, model_save_name)

        model_type = "longlingam"

        model_name_longlingam = "{}_{}_{}_{}_{}".format(N, d, seed_graph, clevel, all_bootstrap_seed,
                                                                    significant_threshold)
        curve_dic = curve_score(curve_dic, B, model_type, 'longitudinal lingam', model_name_longlingam)

        for metric_name_tmp in metric_name_lst:
            for model_name, v in curve_dic.items():
                curve_dic_boot1.setdefault(model_name, {})
                curve_dic_boot1[model_name].setdefault(metric_name_tmp, []).append(v[metric_name_tmp])

    curve_dic_boot2 = {}
    for all_bootstrap_seed in args.all_bootstrap_seed_lst:
        best_result = get_best_result(args, N, d, seed_graph, clevel, all_using_weight,
                                      all_bootstrap_seed,
                                      significant_threshold, golem_gamma_lst=golem_gamma_lst,
                                      notears_gamma_lst=notears_gamma_lst)
        curve_dic = {}
        """-------------------- golem & notears --------------------"""
        for model_name, v in best_result.items():
            model_save_name = v['model name']
            model_type = model_name.split(" ")[0]
            curve_dic = curve_score(curve_dic, B, model_type, model_name, model_save_name)

        model_type = "longlingam"

        model_name_longlingam = "{}_{}_{}_{}_{}".format(N, d, seed_graph, clevel, all_bootstrap_seed,
                                                                    significant_threshold)
        curve_dic = curve_score(curve_dic, B, model_type, 'longitudinal lingam', model_name_longlingam)

        for metric_name_tmp in metric_name_lst:
            for model_name, v in curve_dic.items():
                curve_dic_boot2.setdefault(model_name, {})
                curve_dic_boot2[model_name].setdefault(metric_name_tmp, []).append(v[metric_name_tmp])

    curve_dic_boot = {}
    for k, v in curve_dic_boot1.items():
        if "vanilla" in k:
            curve_dic_boot[k] = v
        else:
            curve_dic_boot[k] = curve_dic_boot2[k]

    plot_figure_add_edges(figure_save_path, B, score_dic_boot, curve_dic_boot, new_name, ori_name,
                          metric_name=metric_name,
                          ylabel=ylabel, xlabel=xlabel, show_figure=show_figure,
                          fig_name="{}_{}_{}_{}_{}".format(metric_name, clevel, seed_graph, N, d))


def get_curve_dic_gamma(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight):
    metric_name_lst = ['fdr', 'tpr', 'trpr', 'fpr', 'shd', 'pred_size', "prec"]
    curve_dic_boot = {}
    for all_bootstrap_seed in args.all_bootstrap_seed_lst:
        best_result = get_result_gamma(args, N, d, seed_graph, all_corrupt_level, all_using_weight,
                                       all_bootstrap_seed)
        curve_dic = {}
        """-------------------- golem & notears --------------------"""
        for model_name, v in best_result.items():
            model_save_name = v['model name']
            model_type = model_name.split(" ")[0]

            cwd = os.getcwd()
            with open(cwd + "/result/{}/result_{}.pkl".format(model_type, model_save_name), "rb") as f:
                result_dic_tmp = pickle.load(f)
            v = result_dic_tmp[" ".join(model_name.split(" ")[:-1])]

            """============== for plotting curve ==========="""
            w_curve = v.copy()
            top_n = len(B) ** 2
            top_n_idx = return_top_n_idx(w_curve, top_n)
            curve_dic.setdefault(model_name, {})
            # curve_lst = []
            for i in range(top_n):
                B_est = np.zeros_like(w_curve)
                for j in range(i + 1):
                    B_est[top_n_idx[j][0], top_n_idx[j][1]] = 1
                result_tmp = count_accuracy(B, B_est)

                # for k, v in result_tmp.items():
                for k2, v2 in result_tmp.items():
                    curve_dic[model_name].setdefault(k2, []).append(v2)
                curve_dic[model_name].setdefault("prec", []).append(1 - result_tmp["fdr"])
        for metric_name in metric_name_lst:

            for model_name, v in curve_dic.items():
                curve_dic_boot.setdefault(model_name, {})
                curve_dic_boot[model_name].setdefault(metric_name, []).append(v[metric_name])
    return curve_dic_boot


def get_result_gamma(args, N, d, seed_graph, all_corrupt_level, all_using_weight, all_bootstrap_seed):
    cwd = os.getcwd()
    best_result = {}
    """Find best model"""
    model_type = "golem"

    for golem_lr in args.golem_lr_lst:
        for golem_gamma in [1, 5, 10, 0.1, 0.01, 0.3]:
            model_name_golem = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                            all_corrupt_level,
                                                                            args.golem_lambda_l1,
                                                                            args.golem_lambda_h,
                                                                            args.golem_lambda_p,
                                                                            golem_lr, args.golem_num_iter,
                                                                            golem_gamma,
                                                                            all_using_weight,
                                                                            all_bootstrap_seed)
            result_dic_golem, log_dic_golem = load_result(cwd, model_type, model_name_golem)
            for k, v in log_dic_golem.items():
                if "soft" in k:
                    name_tmp = "golem soft" + " {}".format(golem_gamma)

                else:
                    name_tmp = k + " {}".format(golem_gamma)
                best_result.setdefault(name_tmp, {"metric": np.inf, "model name": None})
                metric_tmp = v['score'][-1]
                if metric_tmp <= best_result[name_tmp]["metric"]:
                    best_result[name_tmp]["metric"] = metric_tmp
                    best_result[name_tmp]["model name"] = model_name_golem

    model_type = "notears"
    for lamb in args.notears_lamb_lst:
        for notears_gamma in [1, 5, 10, 100, 1000]:
            notears_lambda1, notears_lambda2 = lamb, lamb
            model_name_notears = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                           all_corrupt_level, notears_gamma,
                                                                           args.notears_p1, args.notears_p2,
                                                                           notears_lambda1, notears_lambda2,
                                                                           all_using_weight, all_bootstrap_seed)
            result_dic_notears, log_dic_notears = load_result(cwd, model_type, model_name_notears)
            for k, v in log_dic_notears.items():
                if "soft" in k:
                    name_tmp = "notears soft {}".format(notears_gamma)

                else:
                    name_tmp = k + " {}".format(notears_gamma)
                best_result.setdefault(name_tmp, {"metric": np.inf, "model name": None})
                metric_tmp = v['loss'][-1][-1]
                if metric_tmp <= best_result[name_tmp]["metric"]:
                    best_result[name_tmp]["metric"] = metric_tmp
                    best_result[name_tmp]["model name"] = model_name_notears
    return best_result


def plot_graph_all_graphs(new_name, ori_name,
                          significant_threshold,golem_gamma_lst,notears_gamma_lst,
                          seed_graph=1, N=1000, d=16, all_corrupt_level=0.0, all_using_weight=False,
                          show_figure=True,
                          metric_name="fdr", ylabel="FDR", xlabel="number of included edges"):
    cwd = os.getcwd()
    figure_save_path = cwd + "/figure/"
    args = exp_settings(123)
    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)
    # N = 1000
    # d = 16
    # all_corrupt_level = 0.0
    # all_using_weight = False

    for seed_graph in range(1, 6):
        save_folder = "/{}_{}_{}_{}_{}_{}_{}_{}".format(seed_graph, args.seed_scm, N, d,
                                                        args.followup_min,
                                                        args.followup_max, args.delta_t,
                                                        args.significant_threshold)
        data_load_path = cwd + "/syn_data" + save_folder

        B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
        precedent_matrix_ori = torch.from_numpy(
            np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()

        # score_dic_boot = get_score_dic(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight, threshold=0.05)
        score_dic_boot_tmp = get_score_dic(args, precedent_matrix_ori, B, N, d, seed_graph, all_corrupt_level,
                                           all_using_weight, threshold=0.1)
        score_dic_boot_tmp
        if seed_graph == 1:
            score_dic_boot = score_dic_boot_tmp
        else:
            for model_name_tmp, v_tmp in score_dic_boot_tmp.items():
                for metric_tmp, lst_tmp in v_tmp.items():
                    score_dic_boot[model_name_tmp][metric_tmp] += lst_tmp

    metric_name_lst = ['fdr', 'tpr', 'trpr', 'fpr', 'shd', 'pred_size', "prec"]
    for seed_graph in range(1, 6):
        save_folder = "/{}_{}_{}_{}_{}_{}_{}_{}".format(seed_graph, args.seed_scm, N, d,
                                                        args.followup_min,
                                                        args.followup_max, args.delta_t,
                                                        args.significant_threshold)
        data_load_path = cwd + "/syn_data" + save_folder

        B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)

        curve_dic_boot_tmp = {}
        for all_bootstrap_seed in args.all_bootstrap_seed_lst:
            best_result = get_best_result(args, N, d, seed_graph, all_corrupt_level, all_using_weight,
                                          all_bootstrap_seed,
                                      significant_threshold, golem_gamma_lst=golem_gamma_lst,
                                      notears_gamma_lst=notears_gamma_lst)
            curve_dic = {}
            """-------------------- golem & notears --------------------"""
            for model_name, v in best_result.items():
                model_save_name = v['model name']
                model_type = model_name.split(" ")[0]
                curve_dic = curve_score(curve_dic, B, model_type, model_name, model_save_name)

            model_type = "longlingam"

            model_name_longlingam = "{}_{}_{}_{}".format(N, d, seed_graph, all_bootstrap_seed)
            curve_dic = curve_score(curve_dic, B, model_type, 'longitudinal lingam', model_name_longlingam)

            for metric_name_tmp in metric_name_lst:
                for model_name, v in curve_dic.items():
                    curve_dic_boot_tmp.setdefault(model_name, {})
                    curve_dic_boot_tmp[model_name].setdefault(metric_name_tmp, []).append(v[metric_name_tmp])
        if seed_graph == 1:
            curve_dic_boot = curve_dic_boot_tmp
        else:
            for model_name_tmp, v_tmp in curve_dic_boot_tmp.items():
                for metric_tmp, lst_tmp in v_tmp.items():
                    curve_dic_boot[model_name_tmp][metric_tmp] += lst_tmp

    del score_dic_boot["Precedence Matrix"]
    plot_figure_add_edges(figure_save_path, B, score_dic_boot, curve_dic_boot, new_name, ori_name,
                          metric_name=metric_name,
                          ylabel=ylabel, xlabel=xlabel, show_figure=show_figure,
                          fig_name="all_{}_{}_{}_{}_{}".format(metric_name, all_corrupt_level, seed_graph, N, d))


if __name__ == "__main__":
    import numpy as np
    from read_result import threshold_score, threshold_score_pc, curve_score, get_best_result, n_edges_score, \
        get_score_dic, get_curve_dic, write_table, plot_graph
    from utils import exp_settings, load_result
    import os
    import torch
    from utils import exp_settings, random_scale, create_weight, random_flip, load_result

    ori_name = ["Precedence Matrix", "pc none", "pc forbid", "fges none", "fges forbid", "longitudinal lingam",
                "notears vanilla",
                "notears hard", "notears soft", "golem vanilla", "golem hard", "golem soft"]
    new_name = ["precedence matrix", "pc", "pc forbid", "fges", "fges forbid", "L-lingam", "notears-mlp",
                "notears-mlp hard",
                "notears-mlp soft", "golem", "golem hard", "golem soft"]
    # write_table(ori_name, new_name, d=16, seed_graph=1, N=1000, threshold_lst=(0.01, 0.1),
    #             corrupt_level_lst=(0.0, 0.25))

    """============================ plot graph ================================"""
    significant_threshold = 0.001
    args = exp_settings(123)

    plot_graph_corrupt(args, new_name, ori_name, significant_threshold, (1, 10), (10,),
                       seed_graph=1, N=1000, d=16, all_using_weight=False,
                       show_figure=True, clevel=0.25,
                       metric_name="tpr", ylabel="Recall (TP/(TP + FN))", xlabel="Prediction Size")

    plot_graph_corrupt(args, new_name, ori_name, significant_threshold, (1, 10), (10,),
                       seed_graph=1, N=1000, d=16, all_using_weight=False,
                       show_figure=True, clevel=0.25,
                       metric_name="prec", ylabel="Precision (TP/(TP + FP))", xlabel="Prediction Size")

    """============================ plot weight graph ================================"""
    from matplotlib.legend_handler import HandlerTuple

    cwd = os.getcwd()

    figure_save_path = cwd + "/figure"
    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)
    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    seed_graph = 1
    N = 1000
    d = 16
    all_corrupt_level = 0.0
    all_using_weight = False
    topk = 50
    save_folder = "/{}_{}_{}_{}_{}_{}_{}_{}".format(seed_graph, args.seed_scm, N, d,
                                                    args.followup_min,
                                                    args.followup_max, args.delta_t,
                                                    args.significant_threshold)
    data_load_path = cwd + "/syn_data" + save_folder

    B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
    precedent_matrix_ori = torch.from_numpy(
        np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()

    weight_dic = get_sort_weight_dic(args, B, N, d, 3, all_corrupt_level, all_using_weight, all_bootstrap_seed=1)
    _, ax = plt.subplots(figsize=(8, 6), dpi=300)
    con = 0
    lgd_lst = []
    lgd_name_lst = []
    for k, v in sorted(weight_dic.items()):
        p1 = ax.scatter([v[j] for j in range(len(v)) if j > len(v) - topk],
                        [i for i in range(len(v)) if i > len(v) - topk],
                        alpha=0.5, label=k, s=10,
                        c=color_lst[con], marker='s')
        p2 = ax.scatter([v[j] for j in range(len(v)) if j <= len(v) - topk],
                        [i for i in range(len(v)) if i <= len(v) - topk],
                        alpha=0.5, label=k, s=10,
                        c=color_lst[con], marker='.')
        lgd_lst.append((p1, p2))
        lgd_name_lst.append(k)
        con += 1

    #         plt.legend([(p1,p2)],k)
    #         l = ax.legend([(p1,p2)],k)
    ax.legend(lgd_lst, lgd_name_lst, handler_map={tuple: HandlerTuple(ndivide=None)})
    #     plt.legend()
    plt.xlabel("Absolute value of weights")
    plt.ylabel("Number of weights")
    # plt.title("Cummulative weight distribution")
    plt.show()
    plt.savefig(figure_save_path + "/sorted_weight.pdf")

    """============================ plot weight graph ================================"""
    cwd = os.getcwd()
    figure_save_path = cwd + "/figure/"
    args = exp_settings(123)
    if not os.path.isdir(figure_save_path):
        os.makedirs(figure_save_path)

    seed_graph = 2
    N = 1000
    d = 16
    all_corrupt_level = 0.0
    all_using_weight = False

    save_folder = "/{}_{}_{}_{}_{}_{}_{}_{}".format(seed_graph, args.seed_scm, N, d,
                                                    args.followup_min,
                                                    args.followup_max, args.delta_t,
                                                    args.significant_threshold)
    data_load_path = cwd + "/syn_data" + save_folder

    B = np.loadtxt("{}/B.csv".format(data_load_path), delimiter=',').astype(np.float32)
    precedent_matrix_ori = torch.from_numpy(
        np.loadtxt("{}/precede_matrix.csv".format(data_load_path), delimiter=',')).float()

    metric_name_lst = ['fdr', 'tpr', 'trpr', 'fpr', 'shd', 'pred_size', "prec"]
    curve_dic_boot = get_curve_dic_gamma(args, B, N, d, seed_graph, all_corrupt_level, all_using_weight)

    metric_name = "tpr"
    ylabel = metric_name
    xlabel = "number of included edges"
    show_figure = True
    plot_lst = ['golem vanilla 1', 'golem soft 1', 'golem soft 5', 'golem soft 10',
                'notears vanilla 1', 'notears soft 1', 'notears soft 5', 'notears soft 10']
    #      'golem hard 1', 'notears hard 1',

    plt.figure(figsize=(8, 6), dpi=300)
    # plt.plot([result_dic["precedent matrix"]["tpr"] for i in range(top_n)], "--", label="precedent matrix")
    plt.plot([np.sum(B) for i in range(100)], [i * 0.01 for i in range(100)], "--", label="#True Edges")

    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color']
    con = 0
    for model_name in plot_lst:

        data_tmp = np.array(curve_dic_boot[model_name][metric_name])
        if "hard" in model_name:
            model_name = " ".join(model_name.split(" ")[:2])

        elif "vanilla" in model_name:
            model_name = model_name.split(" ")[0]
        else:
            model_name = " ".join(model_name.split(" ")[:2]) + " gamma=" + model_name.split(" ")[-1]
        mean_tmp = np.mean(data_tmp, axis=0)
        std_tmp = np.std(data_tmp, axis=0)

        plt.plot(mean_tmp, label=model_name, color=color_lst[con])
        plt.fill_between([i for i in range(len(mean_tmp))],
                         [max(i, 0) for i in mean_tmp - std_tmp],
                         [min(i, 1) for i in mean_tmp + std_tmp],
                         alpha=0.1, color=color_lst[con])
        con += 1
    # for k, v in w_dic_point.items():
    #     plt.scatter([np.sum(v)], [result_dic[k]["tpr"]], label=k)
    plt.legend(loc=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_figure:
        plt.show()
    else:
        plt.savefig(figure_save_path + "{}_{}_{}_{}.pdf".format("gamma", metric_name, seed_graph, all_corrupt_level))
    plt.close()
