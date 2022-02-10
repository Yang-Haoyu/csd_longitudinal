from run_helper import run_exp
import os
from utils import save_result, exp_settings
import time
import sys

""" =========== these paras stay unchange ============"""
model_type = "longlingam"

args = exp_settings(123)
seed_scm = args.seed_scm
cwd = os.getcwd()

""" =========== these paras stay unchange ============"""
d = sys.argv[1]
total_iter =  (args.seed_graph_upper - 1)*len(args.N_lst)*len(args.all_bootstrap_seed_lst) * len(args.all_corrupt_level_lst)
con = 0

for significant_threshold in args.significant_threshold_lst:
    for seed_graph in range(1, args.seed_graph_upper):
        for N in args.N_lst:
            for all_bootstrap_seed in args.all_bootstrap_seed_lst:
                for all_corrupt_level in args.all_corrupt_level_lst:

                    con += 1
                    st = time.time()

                    model_name_longlingam = "{}_{}_{}_{}_{}".format(N, d, seed_graph, all_corrupt_level, all_bootstrap_seed,
                                                                    significant_threshold)
                    print("running {}; The setting is {}".format(model_type, model_name_longlingam))

                    """ =========== load data ============"""
                    data_save_folder = "/{}_{}_{}_{}_{}_{}_{}/".format(
                        seed_graph, N, d, args.followup_min, args.followup_max, args.delta_t,
                        significant_threshold)
                    data_load_path = cwd + "/syn_data" + data_save_folder
                    """ =========== run model ============"""
                    result_dic, log_dic = run_exp(args, data_load_path, model_type,
                                                  all_bootstrap_seed=all_bootstrap_seed)

                    """ =========== save result ============"""
                    save_result(cwd, model_type, log_dic, result_dic, model_name_longlingam)

                    end = time.time()
                    running_time = end - st
                    print("{:.2f} h remaining".format(running_time * (total_iter - con) / 3600))