from run_helper import run_exp
import os
from utils import save_result, exp_settings
import sys
import time

""" =========== load data ============"""
model_type = "golem"

args = exp_settings(123)
golem_lambda_l1 = args.golem_lambda_l1
golem_lambda_h = args.golem_lambda_h
golem_lambda_p = args.golem_lambda_p
golem_num_iter = args.golem_num_iter
cwd = os.getcwd()

""" =========== these paras stay unchange ============"""

total_iter = len(args.significant_threshold_lst) * len(args.gamma_lst) * (args.seed_graph_upper - 1) * len(args.N_lst)\
             * len(args.golem_lr_lst) * len(args.all_bootstrap_seed_lst) * len(args.all_using_weight_lst) * len(args.all_corrupt_level_lst)
con = 0

d = sys.argv[1]
significant_threshold = float(sys.argv[2])
golem_gamma = 10

for seed_graph in range(1, args.seed_graph_upper):
    for N in args.N_lst:
        for golem_lr in args.golem_lr_lst:
            for all_bootstrap_seed in args.all_bootstrap_seed_lst:
                for all_using_weight in args.all_using_weight_lst:
                    for all_corrupt_level in args.all_corrupt_level_lst:
                        con += 1
                        st = time.time()

                        model_name_golem = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(N, d, seed_graph,
                                                                                        all_corrupt_level,
                                                                                        all_bootstrap_seed,
                                                                                        significant_threshold,
                                                                                        golem_gamma,
                                                                                        golem_lambda_l1,
                                                                                        golem_lambda_h,
                                                                                        golem_lambda_p,
                                                                                        golem_lr,
                                                                                        golem_num_iter,
                                                                                        all_using_weight)
                        print("running {}; The setting is {}".format(model_type, model_name_golem))

                        """ =========== load data ============"""
                        data_save_folder = "/{}_{}_{}_{}_{}_{}_{}/".format(
                            seed_graph, N, d, args.followup_min, args.followup_max, args.delta_t,
                            significant_threshold)
                        data_load_path = cwd + "/syn_data" + data_save_folder

                        """ =========== run model ============"""
                        result_dic, log_dic = run_exp(args, data_load_path, model_type,
                                                      all_corrupt_level=all_corrupt_level,
                                                      golem_lambda_l1=golem_lambda_l1, golem_lambda_h=golem_lambda_h,
                                                      golem_lambda_p=golem_lambda_p, golem_lr=golem_lr,
                                                      golem_num_iter=golem_num_iter,
                                                      golem_gamma=golem_gamma,
                                                      all_bootstrap_seed=all_bootstrap_seed,
                                                      all_using_weight=all_using_weight)

                        """ =========== save result ============"""
                        save_result(cwd, model_type, log_dic, result_dic, model_name_golem)

                        end = time.time()
                        running_time = end - st
                        print("{:.2f} h remaining".format(running_time * (total_iter - con) / 3600))
