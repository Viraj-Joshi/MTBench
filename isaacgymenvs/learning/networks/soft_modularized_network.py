######## SOME CODE IS COPIED FROM MTRL REPO ########
######## Link:https://github.com/facebookresearch/mtrl/blob/main/mtrl ########

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Implementation of the soft routing network and MLP described in
"Multi-Task Reinforcement Learning with Soft Modularization"
Link: https://arxiv.org/abs/2003.13661
"""
from rl_games.algos_torch import network_builder

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from isaacgymenvs.learning.networks import moe_layer

def weight_init_linear(m: torch.nn.Module):
    assert isinstance(m.weight, torch.Tensor)
    nn.init.xavier_uniform_(m.weight)
    assert isinstance(m.bias, torch.Tensor)
    nn.init.zeros_(m.bias)


def weight_init_conv(m: torch.nn.Module):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert isinstance(m.weight, torch.Tensor)
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)  # type: ignore[operator]
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain("relu")
    assert isinstance(m.weight, torch.Tensor)
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init_moe_layer(m: torch.nn.Module):
    assert isinstance(m.weight, torch.Tensor)
    for i in range(m.weight.shape[0]):
        nn.init.xavier_uniform_(m.weight[i])
    assert isinstance(m.bias, torch.Tensor)
    nn.init.zeros_(m.bias)


def weight_init(m: torch.nn.Module):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        weight_init_linear(m)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        weight_init_conv(m)
    elif isinstance(m, moe_layer.Linear):
        weight_init_moe_layer(m)

class TaskEncoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Task encoder encodes the task embedding though fully connected layers
    """
    def __init__(self, D, **mlp_args):
        super().__init__()
        if len(mlp_args['units']) == 0:
            self.mlp = nn.Sequential(nn.Linear(mlp_args['input_size'],D),nn.ReLU())
        else:
            self.mlp = self._build_mlp(**mlp_args)
            last_layer = list(self.mlp.children())[-2].out_features
            self.mlp = nn.Sequential(*list(self.mlp.children()), nn.Linear(last_layer, D))

    def forward(self, embedding):
        return self.mlp(embedding)

class Encoder(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(self, D, **mlp_args):
        super().__init__()

        self.mlp = self._build_mlp(**mlp_args)
        last_layer = list(self.mlp.children())[-2].out_features # -2 gets the last linearity AND activation
        self.mlp = nn.Sequential(*list(self.mlp.children()), nn.Linear(last_layer, D))

    def forward(self, inp):
        return self.mlp(inp)

class SoftModularizedMLP(torch.nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_layers: int,
        hidden_features: int,
        bias: bool = True,
    ):
        """Class to implement the actor/critic in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        It is similar to layers.FeedForward but allows selection of expert
        at each layer.
        """
        super().__init__()
        
        layers: List[nn.Module] = []
        current_in_features = hidden_features

        for _ in range(num_layers - 1):
            linear = moe_layer.Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=bias,
            )
            layer = nn.Sequential(linear, nn.ReLU())

            layers.append(layer)
            # Each layer is a combination of a moe layer and ReLU.
            current_in_features = hidden_features
        linear = moe_layer.Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=out_features,
            bias=bias,
        )

        layers.append(linear)
        self.layers = nn.ModuleList(layers)
        self.routing_network = RoutingNetwork(
            hidden_features=hidden_features,
            num_layers=num_layers - 1,
            num_experts_per_layer=num_experts,
        )

    """
    Using Fig 2 of https://arxiv.org/pdf/2003.13661
    - f is the encoded state or state_action of shape (B,D)
    - inp IS NOT the observation
      inp IS the element wise multiplication of the encoded 
      task embedding of shape (B,D) and the encoded state f
    """
    def forward(self, f, inp: torch.Tensor) -> torch.Tensor:
        probs = self.routing_network(inp)
        # (num_layers, B, num_experts, num_experts)
        probs = probs.permute(0, 2, 3, 1)
        # (num_layers, num_experts, num_experts, B)
        num_experts = probs.shape[1]
        
        x = inp                                                                     ### IMPORTANT: using inp vs f_obs is a design choice, in the paper they use f_obs ###
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index]
            # (num_experts, num_experts, B)
            x = layer(x)                                                            # After layer transformation
            # (num_experts, B, dim2)
            _out = p.unsqueeze(-1) * x.unsqueeze(0).repeat(num_experts, 1, 1, 1)    # After multiplication with probabilities
            # (num_experts, num_experts, B, dim2)
            x = _out.sum(dim=1)                                                     # After averaging experts
            # (num_experts, batch, dim2)
        out = self.layers[-1](x).sum(dim=0)
        # (B, out_dim)
        return out
    
class SoftModularizedMLPWrapper(SoftModularizedMLP):
    def __init__(self, output_dim, task_encoder_args, state_encoder_args, network_args):
        num_experts = network_args['num_experts']
        in_features = network_args['input_size']
        num_layers = network_args['num_layers']
        hidden_features = network_args['D']

        super().__init__(num_experts, in_features, output_dim, num_layers, hidden_features)

        self.task_encoder = TaskEncoder(hidden_features, **task_encoder_args)
        self.encoder = Encoder(hidden_features, **state_encoder_args)

        self.task_embedding_dim = task_encoder_args['input_size']

    def forward(self, obs):
        true_obs = obs[:,:-self.task_embedding_dim]
        # (B, obs_dim)
        task_embedding = obs[:,-self.task_embedding_dim:]
        # (B, task_embedding_dim)

        f_obs = self.encoder(true_obs)
        # (B, D)
        z = self.task_encoder(task_embedding)
        # (B, D)
        output = super().forward(f_obs, f_obs * z)
        # (B, output_dim)

        return output
    
class SharedPPOSoftModularizedMLP(torch.nn.Module):
    def __init__(
        self,
        actor_output_dim,
        critic_output_dim,
        task_encoder_args,
        state_encoder_args,
        network_args,
    ):
        """Class to implement the actor/critic in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        It is similar to layers.FeedForward but allows selection of expert
        at each layer.
        """
        super().__init__()
        self.task_embedding_dim = task_encoder_args['input_size']
        num_experts = network_args['num_experts']
        num_layers = network_args['num_layers']
        hidden_features = network_args['D']
        unique_task_indices = network_args['unique_task_indices']

        # init TaskEncoder
        self.task_encoder = TaskEncoder(hidden_features, **task_encoder_args)

        # init StateEncoder
        self.state_encoder = Encoder(hidden_features, **state_encoder_args)
        
        layers: List[nn.Module] = []
        current_in_features = hidden_features

        for _ in range(num_layers - 1):
            linear = moe_layer.Linear(
                num_experts=num_experts,
                in_features=current_in_features,
                out_features=hidden_features,
                bias=True,
            )
            layer = nn.Sequential(linear, nn.ReLU())

            layers.append(layer)
            # Each layer is a combination of a moe layer and ReLU.
            current_in_features = hidden_features
        
        self.actor_head = moe_layer.Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=actor_output_dim,
        )
        self.critic_head = moe_layer.Linear(
            num_experts=num_experts,
            in_features=current_in_features,
            out_features=critic_output_dim,
        )

        self.layers = nn.ModuleList(layers)
        self.routing_network = RoutingNetwork(
            hidden_features=hidden_features,
            num_layers=num_layers - 1,
            num_experts_per_layer=num_experts,
        )

    """
    Using Fig 2 of https://arxiv.org/pdf/2003.13661
    inp IS the observation
    """
    def forward(self, inp: torch.Tensor) -> torch.Tensor: 
        obs = inp[:,:-self.task_embedding_dim]
        # (B, obs_dim)
        task_embedding = inp[:,-self.task_embedding_dim:]
        # (B, task_embedding_dim)
        f_obs = self.state_encoder(obs)
        # (B, D)
        z = self.task_encoder(task_embedding)
        # (B, D)
        probs = self.routing_network(f_obs * z)
        # (num_layers, B, num_experts, num_experts)
        probs = probs.permute(0, 2, 3, 1)
        # (num_layers, num_experts, num_experts, B)
        num_experts = probs.shape[1]
        
        x = f_obs
        for index, layer in enumerate(self.layers[:-1]):
            p = probs[index]
            # (num_experts, num_experts, B)
            x = layer(x)                                                            # After layer transformation
            # (num_experts, B, dim2)
            _out = p.unsqueeze(-1) * x.unsqueeze(0).repeat(num_experts, 1, 1, 1)    # After multiplication with probabilities
            # (num_experts, num_experts, B, dim2)
            x = _out.sum(dim=1)                                                     # After averaging experts
            # (num_experts, batch, dim2)
        mu = self.actor_head(x).sum(dim=0) 
        # (batch, action_dim)
        values = self.critic_head(x).sum(dim=0) 
        # (batch, value_size)
        return mu, values
    
class RoutingNetwork(torch.nn.Module):
    def __init__(
        self,
        hidden_features: int,
        num_experts_per_layer: int,
        num_layers: int,
    ) -> None:
        """Class to implement the routing network in
        'Multi-Task Reinforcement Learning with Soft Modularization' paper.
        """
        super().__init__()

        self.num_experts_per_layer = num_experts_per_layer

        # self.projection_before_routing = nn.Linear(
        #     in_features=in_features,
        #     out_features=hidden_features,
        # ) # not needed if you have a separate encoder and task encoder

        self.W_d = nn.ModuleList(
            [
                nn.Linear(
                    in_features=hidden_features,
                    out_features=self.num_experts_per_layer ** 2,
                )
                for _ in range(num_layers)
            ]
        )

        self.W_u = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.num_experts_per_layer ** 2,
                    out_features=hidden_features,
                )
                for _ in range(num_layers - 1)
            ]
        )  # the first layer does not need W_u

        self._softmax = nn.Softmax(dim=2)

    def _process_logprob(self, logprob: torch.Tensor) -> torch.Tensor:
        logprob_shape = logprob.shape
        logprob = logprob.reshape(
            logprob_shape[0], self.num_experts_per_layer, self.num_experts_per_layer
        )
        # logprob[:][i][j] == weight (probability )of the ith module (in current layer)
        # for contributing to the jth module in the next layer.
        # Since the ith module has to divide its weight among all modules in the
        # next layer, logprob[batch_index][i][:] sums to 1
        prob = self._softmax(logprob)
        return prob

    # inp IS the element wise multiplication of the encoded 
    # task embedding and the encoded state f of shapes (B,D)
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp = self.projection_before_routing(inp)
        # (batch, hidden_features)

        p = self.W_d[0](F.relu(inp))  # batch x num_experts ** 2
        prob = [p]
        for W_u, W_d in zip(self.W_u, self.W_d[1:]):
            p = W_d(F.relu((W_u(prob[-1]) * inp)))
            prob.append(p)
        prob_tensor = torch.cat(
            [self._process_logprob(logprob=logprob).unsqueeze(0) for logprob in prob],
            dim=0,
        )
        # (num_layers-1, batch, num_experts, num_experts)
        return prob_tensor
    
import numpy as np
def _gaussian_logprob(noise: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute the gaussian log probability.

    Args:
        noise (torch.Tensor):
        log_std (torch.Tensor): [description]

    Returns:
        torch.Tensor: log-probaility of the sample.
    """
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def _squash(
    mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply squashing function.
        See appendix C from https://arxiv.org/pdf/1812.05905.pdf.

    Args:
        mu ([torch.Tensor]): mean of the gaussian distribution.
        pi ([torch.Tensor]): sample from the gaussian distribution.
        log_pi ([torch.Tensor]): log probability.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of
            (squashed mean of the gaussian, squashed sample from the gaussian,
                squashed  log-probability of the sample)
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi