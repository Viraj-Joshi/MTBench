import copy
import random
from typing import List, Tuple

import torch
import numpy as np
from scipy.optimize import minimize



###############  PCGrad Implemenation ################

def _project_conflicting(grads: List[Tuple[torch.Tensor]]):
    pc_grad = copy.deepcopy(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = sum(
                [
                    torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                    for grad_i, grad_j in zip(g_i, g_j)
                ]
            )
            if g_i_g_j < 0:
                g_j_norm_square = (
                    torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                )
                for grad_i, grad_j in zip(g_i, g_j):
                    grad_i -= g_i_g_j * grad_j / g_j_norm_square

    merged_grad = [sum(g) for g in zip(*pc_grad)]
    # by default use reduction mean
    merged_grad = [g / len(grads) for g in merged_grad]

    return merged_grad

def pcgrad_backward(loss, extra_losses, task_indices, shared_parameters):
    """ Backward pass with PCGrad algorithm to implement gradients. Extra losses
    can be viewed as extra tasks.
    """
    losses = []
    tids = torch.unique(task_indices)
    for tid in tids:
        mask = task_indices == tid
        losses.append(loss[mask].mean())
    losses += extra_losses
    n_tasks = len(losses)

    shared_grads = []
    for l in losses:
        shared_grads.append(
            torch.autograd.grad(l, shared_parameters, retain_graph=True)
        )

    non_conflict_shared_grads = _project_conflicting(shared_grads)

    for p, g in zip(shared_parameters, non_conflict_shared_grads):
        p.grad = g


###############  CAGrad Implemenation ################

def grad2vec(shared_params, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    # for mm in m.shared_modules():
    #     for p in mm.parameters():

    for param in shared_params:
        grad = param.grad
        if grad is not None:
            grad_cur = grad.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
        cnt += 1

def cagrad(grads, alpha=0.5, rescale=1):
    n_tasks = grads.size(1)
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(n_tasks) / n_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1))
            + c
            * np.sqrt(
                x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1))
                + 1e-8
            )
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g, GG.numpy(), w_cpu
    elif rescale == 1:
        return g / (1 + alpha ** 2), GG.numpy(), w_cpu
    else:
        return g / (1 + alpha), GG.numpy(), w_cpu


def overwrite_grad(shared_parameters, newgrad, grad_dims, n_tasks):
    newgrad = newgrad * n_tasks  # to match the sum loss
    cnt = 0

    # for mm in m.shared_modules():
    #     for param in mm.parameters():
    for param in shared_parameters:
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[: cnt + 1])
        this_grad = newgrad[beg:en].contiguous().view(param.data.size())
        param.grad = this_grad.data.clone()
        cnt += 1

def cagrad_backward(loss, extra_losses, task_indices, shared_parameters, c=0.4):
    """ Backward pass with CAGrad algorithm to implement gradients. Extra losses
    can be viewed as extra tasks.
    """
    losses = []
    tids = torch.unique(task_indices)
    for tid in tids:
        mask = task_indices == tid
        losses.append(loss[mask].mean())
    losses += extra_losses
    n_tasks = len(losses)
    
    grad_dims = []
    for param in shared_parameters:
        grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), n_tasks).to(loss.get_device())

    for i in range(n_tasks):
        losses[i].backward(retain_graph=True)
        grad2vec(shared_parameters, grads, grad_dims, i)
        for p in shared_parameters:
            p.grad = None
    
    g, GTG, w_cpu = cagrad(grads, alpha=c, rescale=1)
    overwrite_grad(shared_parameters, g, grad_dims, n_tasks)

    return GTG, w_cpu

###############  FAMO Implemenation ################

def famo_backward(loss, extra_losses, task_indices, shared_parameters, c=0.4):
    """ Backward pass with FAMO algorithm to implement gradients. Extra losses
    can be viewed as extra tasks.
    """
    losses = []
    tids = torch.unique(task_indices)
    for tid in tids:
        mask = task_indices == tid
        losses.append(loss[mask].mean())
    for loss in extra_losses:
        for tid in tids:
            mask = task_indices == tid
            losses.append(loss[mask].mean())
    import ipdb; ipdb.set_trace()
    # losses += extra_losses
    n_tasks = len(losses)
    
    grad_dims = []
    for param in shared_parameters:
        grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), n_tasks).to(loss.get_device())

    for i in range(n_tasks):
        losses[i].backward(retain_graph=True)
        grad2vec(shared_parameters, grads, grad_dims, i)
        for p in shared_parameters:
            p.grad = None
    
    g, GTG, w_cpu = cagrad(grads, alpha=c, rescale=1)
    overwrite_grad(shared_parameters, g, grad_dims, n_tasks)

    return GTG, w_cpu