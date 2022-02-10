from run_helper import run_exp
import os

from utils import save_result, exp_settings
import sys

import time

""" =========== load data ============"""
model_type = "notears"

d = int(sys.argv[1])
N = int(sys.argv[2])
seed_graph = int(sys.argv[3])

args = exp_settings(123)

notears_gamma = 10
notears_p1 = args.notears_p1
notears_p2 = args.notears_p2

total_iter = len(args.significant_threshold_lst)*len(args.notears_lamb_lst) * \
             len(args.all_bootstrap_seed_lst) * len(args.all_using_weight_lst) * len(args.all_corrupt_level_lst)
con = 0
for significant_threshold in args.significant_threshold_lst:
    for notears_lamb in args.notears_lamb_lst:
        for all_bootstrap_seed in args.all_bootstrap_seed_lst:
            for all_using_weight in args.all_using_weight_lst:
                for all_corrupt_level in args.all_corrupt_level_lst:
                    con += 1
                    st = time.time()

                    notears_lambda1, notears_lambda2 = notears_lamb, notears_lamb
                    model_name_notears = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                                      all_corrupt_level,
                                                                                      all_bootstrap_seed,
                                                                                      significant_threshold,
                                                                                      notears_gamma,
                                                                                      notears_p1, notears_p2,
                                                                                      notears_lambda1,
                                                                                      notears_lambda2,
                                                                                      all_using_weight)
                    print("running {}; The setting is {}".format(model_type, model_name_notears))

                    cwd = os.getcwd()

                    """ =========== load data ============"""
                    data_save_folder = "/{}_{}_{}_{}_{}_{}_{}/".format(
                        seed_graph, N, d, args.followup_min, args.followup_max, args.delta_t,
                        significant_threshold)
                    data_load_path = cwd + "/syn_data" + data_save_folder

                    """ =========== run model ============"""
                    result_dic, log_dic = run_exp(args, data_load_path, model_type,
                                                  all_corrupt_level=all_corrupt_level,
                                                  notears_gamma=notears_gamma, notears_p1=notears_p1,
                                                  notears_p2=notears_p2, notears_lambda1=notears_lambda1,
                                                  notears_lambda2=notears_lambda2,
                                                  all_bootstrap_seed=all_bootstrap_seed,
                                                  all_using_weight=all_using_weight)

                    """ =========== save result ============"""
                    save_result(cwd, model_type, log_dic, result_dic, model_name_notears)

                    end = time.time()
                    running_time = end - st
                    print("{:.2f} h remaining".format(running_time * (total_iter - con) / 3600))
