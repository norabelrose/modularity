# *
# Taken with modification from the PyHessian library.
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian. If not, see <http://www.gnu.org/licenses/>.
# *

from tqdm.auto import tqdm
import torch as th
import numpy as np


# Utility functions
def group_product(xs: list[th.Tensor], ys: list[th.Tensor]) -> th.Tensor:
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    assert xs and ys
    return th.stack([th.sum(x * y) for x, y in zip(xs, ys)]).sum()


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v) ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0.0 if param.grad is None else param.grad + 0.0)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = th.autograd.grad(
        gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True
    )
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


class Hessian:
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (
            data == None and dataloader != None
        )

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        self.device = "cuda" if cuda else "cpu"

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if cuda:
                self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [th.zeros_like(p) for p in self.params]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = th.autograd.grad(
                gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=False
            )
            THv = [THv1 + Hv1 * float(tmp_num_data) + 0.0 for THv1, Hv1 in zip(THv, Hv)]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def topk_eig(self, k: int = 1, max_iter=100, tol=1e-3) -> tuple[list, list]:
        """
        Compute the top k eigenvalues and eigenvectors using power iteration.

        Args:
            k: Number of eigenvalues to be computed.
            max_iter: maximum iterations used to compute each single eigenvalue
            tol: the relative tolerance between two consecutive eigenvalue
                computations from power iteration
        """

        assert k >= 1

        eigenvalues = []
        eigenvectors = []

        for i in range(k):
            if k > 1:
                print(f"Computing eigenvalue {i + 1} of {k}")

            eigenvalue = None
            v = [th.randn_like(p) for p in self.params]  # Random vector in weight space
            v = normalization(v)  # normalize the vector

            pbar = tqdm(range(max_iter), desc="Power iteration")
            for i in pbar:
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tol
                    ):
                        pbar.close()
                        break
                    else:
                        eigenvalue = tmp_eigenvalue

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

        return eigenvalues, eigenvectors

    def trace(self, max_iter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        max_iter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.0

        for i in range(max_iter):
            self.model.zero_grad()
            v = [th.randint_like(p, high=2, device=device) for p in self.params]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [th.randint_like(p, high=2) for p in self.params]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initialization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in tqdm(range(iter), desc="Lanczos quadrature"):
                self.model.zero_grad()
                w_prime = [th.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = th.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.0:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [th.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = th.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = th.linalg.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = b_[0, :] ** 2
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
