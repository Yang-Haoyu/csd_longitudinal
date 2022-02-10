from models.notears.locally_connected import LocallyConnected
from models.notears.lbfgsb_scipy import LBFGSBScipy
# from locally_connected import LocallyConnected
# from lbfgsb_scipy import LBFGSBScipy
import torch
import torch.nn as nn
import numpy as np
from models.notears.utils import weighted_squared_loss, weighted_bce_loss, squared_loss, bce_loss


class NotearsMLP_precedent(nn.Module):
    def __init__(self, dims, precedent_matrix, bias=True, p_term="hard", gamma=1, p1=1, p2=1):
        super(NotearsMLP_precedent, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        self.precedent_matrix = precedent_matrix
        self.p_term = p_term
        self.p1 = p1
        self.p2 = p2
        self.gamma = gamma
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        if self.p_term == "hard":
            """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
            d = self.dims[0]
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
            fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
            A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
            h = torch.norm(A * (1 - self.precedent_matrix))

        # elif self.p_term == "soft":
        #     d = self.dims[0]
        #     fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        #     fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        #     A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        #     # h = trace_expm(A) - d  # (Zheng et al. 2018)
        #     M = torch.eye(d) + A / d  # (Yu et al. 2019)
        #     E = torch.matrix_power(M, d - 1)
        #     h1 = (E.t() * M).sum() - d
        #     h2 = torch.exp(-torch.norm(A * self.precedent_matrix))
        #     h = h1 + h2
        elif self.p_term == "soft":
            d = self.dims[0]
            fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
            fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
            A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
            # h = trace_expm(A) - d  # (Zheng et al. 2018)
            M = torch.eye(d) + A / d  # (Yu et al. 2019)
            E = torch.matrix_power(M, d - 1)
            h1 = (E.t() * M).sum() - d
            h2 = torch.abs(
                torch.norm(A * self.precedent_matrix, p=self.p1) - self.gamma * torch.norm(self.precedent_matrix,
                                                                                           p=self.p2))
            h = h1 + h2
        else:
            print("Please choose p_term from [\"None\", \"soft\", \"hard\"]")
            raise NotearsMLP_precedent
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        M = torch.eye(d) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def dual_ascent_step(model, x_input, x_reconstr, lambda1,
                     lambda2, rho, alpha, h, rho_max,
                     weight_matrix=None, loss_type="mse"):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    h_new_lst = []
    loss_lst = []
    rho_lst = []
    alpha_lst = []
    w_lst = []
    optimizer = LBFGSBScipy(model.parameters())
    # X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(x_input)

            if weight_matrix is None:
                if loss_type == "mse":
                    loss = squared_loss(X_hat, x_reconstr)
                elif loss_type == "bce":
                    loss = bce_loss(X_hat, x_reconstr)
                else:
                    raise NotImplementedError

            else:
                if loss_type == "mse":
                    loss = weighted_squared_loss(X_hat, x_reconstr, weight_matrix)
                elif loss_type == "bce":
                    loss = weighted_bce_loss(X_hat, x_reconstr, weight_matrix)
                else:
                    raise NotImplementedError

            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            W_est = model.fc1_to_adj()
            h_new = model.h_func().item()
            h_new_lst.append(h_new)
            X_hat = model(x_input)

            if weight_matrix is None:
                if loss_type == "mse":
                    loss = squared_loss(X_hat, x_reconstr)
                elif loss_type == "bce":
                    loss = bce_loss(X_hat, x_reconstr)
                else:
                    raise NotImplementedError

            else:
                if loss_type == "mse":
                    loss = weighted_squared_loss(X_hat, x_reconstr, weight_matrix)
                elif loss_type == "bce":
                    loss = weighted_bce_loss(X_hat, x_reconstr, weight_matrix)
                else:
                    raise NotImplementedError

            loss_lst.append(loss.item())
            rho_lst.append(rho)
            alpha_lst.append(alpha)
            w_lst.append(W_est)

        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new, h_new_lst, loss_lst, rho_lst, alpha_lst, w_lst


def run_notears_mlp(x_input: np.ndarray, x_reconstr: np.ndarray, precedent_matrix: np.ndarray,
                    weight_matrix = None,
                    loss_type = "mse", p_term = "hard", gamma=1, p1=1, p2=1,
                    lambda1: float = 0., lambda2: float = 0., max_iter: int = 100,
                    h_tol: float = 1e-8, rho_max: float = 1e+16):
    d = x_input.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if p_term == "None":
        print("building NOTEARS-MLP ...")
        model = NotearsMLP(dims=[d, 2, 1], bias=True).to(device)

    else:
        print("building NOTEARS-MLP w/ precedent matrix {} ...".format(p_term))
        model = NotearsMLP_precedent(dims=[d, 2, 1], precedent_matrix=precedent_matrix, bias=True, p_term=p_term,
                                     gamma=gamma, p1=p1, p2=p2).to(device)

    rho, alpha, h = 1.0, 0.0, np.inf
    log_dic = {"h": [], "loss": [], "rho": [], "alpha": [], "w": []}
    for epoch in range(max_iter):
        rho, alpha, h, h_lst, loss_lst, rho_lst, alpha_lst, w_lst = \
            dual_ascent_step(model, x_input, x_reconstr, lambda1, lambda2, rho, alpha, h, rho_max,
                             weight_matrix, loss_type=loss_type)

        log_dic["h"].append(h_lst)
        log_dic["loss"].append(loss_lst)
        log_dic["rho"].append(rho_lst)
        log_dic["alpha"].append(alpha_lst)
        log_dic["w"].append(w_lst)
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.fc1_to_adj()
    # W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, log_dic
