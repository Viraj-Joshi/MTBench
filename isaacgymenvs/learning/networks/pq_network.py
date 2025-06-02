import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from typing import List

from rl_games.algos_torch import network_builder

class QNetwork(network_builder.NetworkBuilder.BaseNetwork):
    def __init__(
        self,
        task_encoder_args,
        is_residual_network: bool,
        action_space_type: str,
        num_bins_per_dim: int,
        in_features: int,
        output_dim: int,
        num_blocks: int,
        hidden_features: int,
        norm_input: bool,
        unique_task_indices: torch.Tensor,
    ):
        """Class to implement the Q network (which is the critic)
        from SOLVING CONTINUOUS CONTROL VIA Q-LEARNING by Seyde et al.
        """
        super().__init__()

        self.in_features = in_features  
        self.output_dim = output_dim
        self.unique_task_indices = unique_task_indices
        self.action_space_type = action_space_type
        
        ### if the setting is multi_discrete, this is the number of bins for each action dimension
        if action_space_type == "multi_discrete":
            self.num_bins_per_dim = num_bins_per_dim
       

        """
        the Q is modeled after the LayerNormAndResidualMLP class in ACME
        https://github.com/google-deepmind/acme/blob/5e7503f95fb7cd9511c34f00f953b4e18c8db6cf/acme/tf/networks/continuous.py#L109
        """
        
        Q : List[nn.Module] = []   
        # projection layer
        if in_features != hidden_features:
            projection = nn.Linear(in_features, hidden_features)
            torch.nn.init.orthogonal_(projection.weight, np.sqrt(2))
            torch.nn.init.constant_(projection.bias, 0)
            Q.extend([
                projection,
                nn.ELU(),
                nn.LayerNorm(hidden_features) if not norm_input else nn.BatchNorm1d(hidden_features),
            ])

        # Add (residual) blocks
        for _ in range(num_blocks-1):
            Q.append(Block(
                hidden_features,
                is_residual_network,
                num_layers_in_block=2
            ))

        # output layer slightly different from Seyde
        Q.append(nn.ELU())
        Q.append(nn.Linear(hidden_features, output_dim))
        torch.nn.init.orthogonal_(Q[-1].weight, np.sqrt(2))
        torch.nn.init.constant_(Q[-1].bias, 0)

        self.Q = nn.Sequential(*Q)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if torch.isnan(obs).any():
            raise ValueError("NaN in real_obs")

        x = self.Q(obs)
        if self.action_space_type == "multi_discrete":
            # forward through Q and reshape network output of shape [B,n_a*n_b] to [B, n_a, n_b]
            action_dim = self.output_dim // self.num_bins_per_dim
            return x.view(-1,action_dim,self.num_bins_per_dim) # split to predict decoupled state-action utilities
        else:
            # just get the Q values for each action
            return x

"""
the following is modeled after the ResidualLayerNormWrapper
from https://github.com/google-deepmind/acme/blob/5e7503f95fb7cd9511c34f00f953b4e18c8db6cf/acme/tf/networks/continuous.py#L79
but we place layernorm after each activation, instead of only at the end of the block
"""
class Block(nn.Module):
    def __init__(self, hidden_features: int, use_residual: bool, num_layers_in_block: int = 2):
        """Residual block with configurable number of linear layers and skip connection."""
        super().__init__()
        
        self.use_residual = use_residual
        
        # note there is no activation after the last linear layer 
        layers = []
        for i in range(num_layers_in_block):
            layers.extend([
                nn.Linear(hidden_features, hidden_features),
                nn.ELU() if i < num_layers_in_block - 1 else nn.Identity(),  # No ReLU after last layer before residual
                nn.LayerNorm(hidden_features) if i < num_layers_in_block - 1 else nn.Identity(),  # No normalization after last layer before residual
            ])
            # Initialize weights
            torch.nn.init.orthogonal_(layers[-3].weight, np.sqrt(2))
            torch.nn.init.constant_(layers[-3].bias, 0)
        
        self.layers = nn.Sequential(*layers)
        self._layer_norm = nn.LayerNorm(hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through all layers
        out = self.layers(x)
        
        # Add residual connection if enabled
        if self.use_residual:
            out = out + x
        
        out = self._layer_norm(out)
        return out

    
class TaskEncoder(network_builder.NetworkBuilder.BaseNetwork):
    """
    Task encoder encodes the task embedding though a MLP.
    """
    def __init__(self, output_dim, **mlp_args):
        super().__init__()
        # if len(mlp_args['units']) == 0:
        #     self.mlp = nn.Sequential(nn.Linear(mlp_args['embedding_dim'],output_dim,nn.ReLU()))
        # else:
        #     self.mlp = self._build_mlp(**mlp_args)
        #     last_layer = list(self.mlp.children())[-2].out_features
        #     self.mlp = nn.Sequential(*list(self.mlp.children()), nn.Linear(last_layer, output_dim))
        self.mlp = torch.nn.Linear(mlp_args['input_size'], output_dim, bias=False)

    def forward(self, embedding):
        return self.mlp(embedding)