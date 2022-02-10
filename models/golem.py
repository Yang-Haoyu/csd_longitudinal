"""
Modified from https://github.com/ignavierng/golem
"""

import torch
from torch import nn
import torch.optim as optim
import numpy as np

class golem(nn.Module):

    def __init__(self, n, d, lambda_1, lambda_2, lambda_3, p_term="hard", equal_variances=False,
                 dagness="zheng", prec_mat=None, p1 = 1, p2 = 1, gamma = 1, weight_mat = None):
        super(golem, self).__init__()
        self.n = n
        self.d = d
        self.lambda_1, self.lambda_2, self.lambda_3 = lambda_1, lambda_2, lambda_3
        self.equal_variances = equal_variances
        self.W = nn.Parameter(torch.empty(d, d))
        torch.nn.init.xavier_uniform_(self.W)
        self.p1 = p1; self.p2 = p2; self.gamma = gamma

        self.p_term = p_term
        self.dagness = dagness
        self.prec_mat = prec_mat
        self.p_his = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weight_mat = weight_mat

    def _compute_likelihood(self, _input, _reconstr):
        ## this works using torch 1.10 + version
        # if self.equal_variances:  # Assuming equal noise variances
        #     return 0.5 * self.d * torch.log(
        #         torch.square(
        #             torch.norm(_reconstr - _input @ self.W)
        #         )
        #     ) - torch.linalg.slogdet(torch.eye(self.d) - self.W)[1]
        # else:  # Assuming non-equal noise variances
        #     return 0.5 * torch.sum(
        #         torch.log(
        #             torch.sum(
        #                 torch.square(self._reconstr - self._input @ self.W), dim=0
        #             )
        #         )
        #     ) - torch.linalg.slogdet(torch.eye(self.d) - self.W)[1]

        if self.weight_mat is None:
            if self.equal_variances:  # Assuming equal noise variances
                return 0.5 * self.d * torch.log(
                    torch.norm(_reconstr - _input @ self.W) ** 2
                ) - torch.slogdet(torch.eye(self.d).to(self.device) - self.W)[1]
            else:  # Assuming non-equal noise variances
                return 0.5 * torch.sum(
                    torch.log(
                        torch.sum(
                            (_reconstr - _input @ self.W) ** 2, dim=0
                        )
                    )
                ) - torch.slogdet(torch.eye(self.d).to(self.device) - self.W)[1]
        else:
            if self.equal_variances:  # Assuming equal noise variances
                return 0.5 * self.d * torch.log(
                    torch.norm(self.weight_mat*(_reconstr - _input @ self.W)) ** 2
                ) - torch.slogdet(torch.eye(self.d).to(self.device) - self.W)[1]
            else:  # Assuming non-equal noise variances
                return 0.5 * torch.sum(
                    torch.log(
                        torch.sum(
                            (self.weight_mat*(_reconstr - _input @ self.W)) ** 2, dim=0
                        )
                    )
                ) - torch.slogdet(torch.eye(self.d).to(self.device) - self.W)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.W, p=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
        """
        if self.dagness == "zheng":
            # (Zheng et al. 2018)
            return torch.trace(torch.matrix_exp(self.W * self.W)) - self.d
        elif self.dagness == "yu":
            # (Yu et al. 2019)
            M = torch.eye(self.d).to(self.device) + (self.W * self.W) / np.sqrt(self.d)
            return torch.trace(torch.matrix_power(M, self.d)) - self.d
        else:
            print("Choose dagness from [\"zheng\", \"yu\"]")
            raise NotImplementedError

    def _compute_p_term(self):
        if self.p_term == "None":
            p_loss = 0
        elif self.p_term == "hard":
            p_loss = torch.norm(self.W * (1 - self.prec_mat), p=1)
        elif self.p_term == "soft":
            p_loss = torch.abs(torch.norm(torch.mul(self.W, self.prec_mat), p=self.p1) - self.gamma * torch.norm(self.prec_mat, p=self.p2))
        else:
            raise NotImplementedError
        return p_loss

    def forward(self, _input, _reconstr, niter):

        self.likelihood = self._compute_likelihood(_input, _reconstr)
        self.L1_penalty = self._compute_L1_penalty()
        self.p_loss = self._compute_p_term()
        self.h = self._compute_h()

        if self.p_loss > 0.99 * self.p_his and self.lambda_3 <1e1:
            self.lambda_3 = 10 * self.lambda_3
        self.p_his = self.p_loss
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h + self.lambda_3 * self.p_loss


class golem_trainer:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""

    def __init__(self, learning_rate=1e-3):
        """Initialize self.

        Args:
            learning_rate (float): Learning rate of Adam optimizer.
                Default: 1e-3.
        """
        self.learning_rate = learning_rate
        self._logger = {}

    def train(self, model, _input, _reconstr, num_iter):
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        W_est = None
        for i in range(0, int(num_iter)):
            score, likelihood, h, W_est, p_loss = self.train_iter(model, _input, _reconstr, i)
            if p_loss == 0:
                self.train_checkpoint(i, score.item(), likelihood.item(), h.item(), 0)
            else:
                self.train_checkpoint(i, score.item(), likelihood.item(), h.item(), p_loss.item())
        return W_est

    def eval_iter(self, model, _input, _reconstr):
        """Evaluation for one iteration. Do not train here.
        """
        model.eval()
        with torch.no_grad():
            model(_input, _reconstr)

        return model.score, model.likelihood, model.h, model.W, model.p_loss

    def train_iter(self, model, _input, _reconstr, niter):
        """Training for one iteration.
        """
        model.train()
        model(_input, _reconstr, niter)
        self.optimizer.zero_grad()
        model.score.backward()
        self.optimizer.step()

        return model.score, model.likelihood, model.h, model.W, model.p_loss

    def train_checkpoint(self, i, score, likelihood, h, p_loss):

        self._logger.setdefault("score",[]).append(score)
        self._logger.setdefault("likelihood",[]).append(likelihood)
        self._logger.setdefault("h",[]).append(h)
        self._logger.setdefault("p_loss",[]).append(p_loss)


def run_golem(x1_imput, x2_imput, precedent_matrix, lambda_1, lambda_2, lambda_3, equal_variances=True,
              num_iter=1e+5, learning_rate=1e-3, dagness="yu", p_term="hard", p1 = 1, p2 = 1, gamma = 1,
              weight_matrix = None):
    # Set up model
    print("running golem - {}".format(p_term))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _input, _reconstr = torch.from_numpy(x1_imput).to(device), torch.from_numpy(x2_imput).to(device)

    n, d = _input.shape
    if weight_matrix is not None:
        weight_matrix = weight_matrix.to(device)
    model = golem(n, d, lambda_1, lambda_2, lambda_3, p_term=p_term, equal_variances=equal_variances,
                  dagness=dagness, prec_mat=precedent_matrix.to(device),p1 = p1, p2 = p2, gamma = gamma,
                  weight_mat = weight_matrix)
    model.to(device)
    # Training
    trainer = golem_trainer(learning_rate)
    W_est = trainer.train(model, _input, _reconstr, num_iter)

    return W_est, trainer._logger, model  # Not thresholded yet
