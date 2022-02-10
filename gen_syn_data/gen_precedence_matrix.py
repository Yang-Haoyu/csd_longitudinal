import itertools
import numpy as np
import warnings


def gen_p(x_mask, d, t_init=60, delta_t=24, significant_threshold=0.01, max_threshold=100):
    # create 2 cross section
    df_t1 = x_mask[:, t_init:t_init + delta_t, :]
    df_t2 = x_mask[:, t_init + delta_t:t_init + delta_t * 2, :]

    precede_list = []
    con = 0
    fea_pair_list = list(itertools.combinations(list(range(d)), 2))

    # loop over all feature pairs
    for event_1, event_2 in fea_pair_list:
        con += 1
        # if among patients who have both A and B in the second time window,
        # B is significantly more likely to be incident than A.
        # select patients who have both A and B in the second time window
        pat_id_intersec = (np.sum(np.abs(df_t2[:, :, event_1]), axis=1) != 0) & (
                np.sum(np.abs(df_t2[:, :, event_2]), axis=1) != 0)

        # choose the first time window
        df_t1_intersec = df_t1[pat_id_intersec]
        df_t1_intersec = df_t1_intersec[:, :, [event_1, event_2]]

        # aggregate the value along the timeline
        df_agg = np.sum(np.abs(df_t1_intersec), axis=1)
        # count the number of two variable in the first time window
        count = np.sum(df_agg != 0, axis=0)

        # if this is significant
        score = np.abs(count[0] - count[1]) / np.max([count[0], count[1]])
        if score > significant_threshold:
            # if the count is greater than max_threshold
            if np.max([count[0], count[1]]) > max_threshold:
                if count[0] > count[1]:
                    # if event_1 is more than event_2
                    # event_1 precede event_2
                    precede_list.append([event_1, event_2, score])
                else:
                    # if event_2 is more than event_1
                    # event_2 precede event_1
                    precede_list.append([event_2, event_1, score])

    # create precedence matrix
    precede_matrix = np.zeros((d, d))
    precede_matrix_soft = np.zeros((d, d))
    # set the value accordingly
    for i, j, score in precede_list:
        precede_matrix[i][j] = 1
        precede_matrix_soft[i][j] = score

    return precede_matrix, precede_matrix_soft, df_t1, df_t2


def get_x_agg(df_t1, df_t2):
    # aggregate the time series into two crosssection
    df_t1[df_t1 == 0] = np.nan
    df_t2[df_t2 == 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x1 = np.nanmean(df_t1, axis=1)
        x2 = np.nanmean(df_t2, axis=1)
    # get the mean value for further imputation
    x_mean = np.nanmean(np.concatenate([df_t1, df_t1], axis=1), axis=(0, 1))
    return x1, x2, x_mean
