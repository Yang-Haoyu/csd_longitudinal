import lingam
import numpy as np


def run_Llingam(x1_imput, x2_imput, precedent_matrix = None):

    # Set up model
    X_t = np.array([x1_imput, x2_imput])
    model = lingam.LongitudinalLiNGAM(n_lags=1)
    model.fit(X_t)

    return model.adjacency_matrices_[1][1], model.adjacency_matrices_ # Not thresholded yet

