import numpy as np
from syn_utils import sigmoid


def follow_time(rng, severity, lamb_min, lamb_max, dist="exp"):
    """
    Calculate the next time that the patient will come back, given certain severity
    :param severity:
    :param lamb_min:
    :param lamb_max:
    :param dist:
    :return:
    """
    if dist == "exp":
        m = np.ceil(rng.exponential(1 - severity)) - 1
        m[m > lamb_min] = lamb_min
        return m.astype(int)
    elif dist == "linear":
        base = lamb_min
        delta = (lamb_max - lamb_min) * (1 - severity)
        noise = rng.uniform(0, (lamb_max - lamb_min) * (1 - severity) / 8)
        corf = (rng.binomial(1, 0.5) - 0.5) * 2
        return np.floor(base + delta + noise * corf).astype(int)
    else:
        raise NotImplementedError

def gen_obs_mask(rng, x, topo_lst, d, N, time_steps, followup_min = 6, followup_max = 24):

    # rearrange x according to topological order
    x_rearrange = x[:,:,topo_lst] # N x t x d

    # random weight to generate severity score
    w_score = rng.random((d,1))
    w_score = np.array([np.exp((w_score[i]*i)/10) for i in range(d)])

    # compute severity score
    s_score =  sigmoid(x_rearrange @ w_score)
    s_score_fea = sigmoid(x)

    # compute the initial encounter
    init_obs_time = follow_time(rng, s_score[:,0,0], followup_min, followup_max, dist = "exp")

    # compute the follow up time
    # follow_up_obs_table = follow_time(s_score, followup_min, followup_max, dist = "linear")
    follow_up_obs_table_fea = follow_time(rng, s_score_fea, followup_min, followup_max, dist = "linear")

    # compute the observation time for each patient
    observation_mask = np.zeros_like(x)
    observation_time_dic = {}
    severity_lag_one_time_dic = {}
    x_lag_one_time_dic = {}
    for pat_id in range(N):
        # compute current observation time
        for fea_num in range(d):
            current_time = init_obs_time[pat_id]
            observation_time_dic.setdefault(fea_num, {})
            x_lag_one_time_dic.setdefault(fea_num, {})
            severity_lag_one_time_dic.setdefault(fea_num, {})
            prev_time = current_time
            while current_time < time_steps:
                observation_time_dic[fea_num].setdefault(pat_id,[]).append(current_time)
                x_lag_one_time_dic[fea_num].setdefault(pat_id, []).append(x[pat_id,prev_time,fea_num])
                severity_lag_one_time_dic[fea_num].setdefault(pat_id, []).append(s_score_fea[pat_id,prev_time,fea_num])
                observation_mask[pat_id, current_time, fea_num] = np.ones(1)
                prev_time = current_time
                # next observation time is based on follow_up_obs_table
                current_time = current_time + int(follow_up_obs_table_fea[pat_id][current_time][fea_num])
    log_lst = [s_score, s_score_fea, observation_time_dic, x_lag_one_time_dic, severity_lag_one_time_dic]
    return observation_mask, log_lst