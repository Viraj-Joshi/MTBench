import torch
import numpy as np
from scipy.optimize import minimize

from isaacgymenvs.learning.mt_sac_agent import MTSACAgent

class CAGradSAC(MTSACAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.c = params.get("c", 0.4)

    def update_critic(self, obs, action, reward, next_obs, not_done, step, task_indices):
        with torch.no_grad():
            dist = self.model.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.model.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha(task_indices) * log_prob

            target_Q = reward + (not_done * self.gamma * target_V)
            target_Q = target_Q.detach()
        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(obs, action)
        critic1_loss = (current_Q1 - target_Q).pow(2)
        critic2_loss = (current_Q2 - target_Q).pow(2)
        # import ipdb; ipdb.set_trace()

        tids = torch.unique(task_indices)
        c_losses = []
        for tid in tids:
            tid_mask = task_indices == tid
            critic_loss = critic1_loss[tid_mask].mean() + critic2_loss[tid_mask].mean()
            c_losses.append(critic_loss)
        params = []
        for param in self.model.sac_network.critic.parameters():
            params.append(param)
        self.critic_optimizer.zero_grad()  # set_to_none=True)
        self.get_weighted_loss(c_losses, params)
        self.critic_optimizer.step()

        return (critic1_loss + critic2_loss).mean(), critic1_loss.mean().detach(), critic2_loss.mean().detach()
    
    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """
        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :
        Returns
        -------
        """
        # NOTE: we allow only shared params for now. Need to see paper for other options.
        n_tasks = len(losses)
        grad_dims = []
        for param in shared_parameters:
            grad_dims.append(param.data.numel())
        grads = torch.Tensor(sum(grad_dims), n_tasks).to(self.device)

        for i in range(n_tasks):
            if i < n_tasks:
                losses[i].mean().backward(retain_graph=True)
            else:
                losses[i].mean().backward()
            self.grad2vec(shared_parameters, grads, grad_dims, i)
            # multi_task_model.zero_grad_shared_modules()
            for p in shared_parameters:
                p.grad = None

        g, GTG, w_cpu = self.cagrad(grads, alpha=self.c, rescale=1)
        self.overwrite_grad(shared_parameters, g, grad_dims, n_tasks)
        return GTG, w_cpu

    def cagrad(self, grads, alpha=0.5, rescale=1):
        n_tasks = grads.shape[1]
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

    @staticmethod
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

    def overwrite_grad(self, shared_parameters, newgrad, grad_dims, n_tasks=1):
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