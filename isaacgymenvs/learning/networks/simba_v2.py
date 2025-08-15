import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def l2normalize(
    tensor: torch.Tensor, axis: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """Computes L2 normalization of a tensor."""
    return tensor / (torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True) + eps)


class Scaler(nn.Module):
    """
    A learnable scaling layer.
    """

    def __init__(
        self,
        dim: int,
        init: float = 1.0,
        scale: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), init * scale, device=device))
        self.forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler.to(x.dtype) * self.forward_scaler * x


class HyperDense(nn.Module):
    """
    A dense layer without bias and with orthogonal initialization.
    """

    def __init__(self, in_dim: int, hidden_dim: int, device: torch.device = None):
        super().__init__()
        self.w = nn.Linear(in_dim, hidden_dim, bias=False, device=device)
        nn.init.orthogonal_(self.w.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


class HyperMLP(nn.Module):
    """
    A small MLP with a specific architecture using HyperDense and Scaler.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8,
        device: torch.device = None,
    ):
        super().__init__()
        self.w1 = HyperDense(in_dim, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.w2 = HyperDense(hidden_dim, out_dim, device=device)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` is required to prevent zero vector.
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperEmbedder(nn.Module):
    """
    Embeds input by concatenating a constant, normalizing, and applying layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
        device: torch.device = None,
    ):
        super().__init__()
        # The input dimension to the dense layer is in_dim + 1
        self.w = HyperDense(in_dim + 1, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.c_shift = c_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_axis = torch.full(
            (*x.shape[:-1], 1), self.c_shift, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, new_axis], dim=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperLERPBlock(nn.Module):
    """
    A residual block using Linear Interpolation (LERP).
    """

    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
        device: torch.device = None,
    ):
        super().__init__()
        self.mlp = HyperMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            out_dim=hidden_dim,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
            device=device,
        )
        self.alpha_scaler = Scaler(
            dim=hidden_dim,
            init=alpha_init,
            scale=alpha_scale,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        mlp_out = self.mlp(x)
        # The original paper uses (x - residual) but x is the residual here.
        # This is interpreted as alpha * (mlp_output - residual_input)
        x = residual + self.alpha_scaler(mlp_out - residual)
        x = l2normalize(x, axis=-1)
        return x


class HyperTanhPolicy(nn.Module):
    """
    A policy that outputs a Tanh action.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.mean_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.mean_w2 = HyperDense(hidden_dim, action_dim, device=device)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mean path
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias.to(mean.dtype)
        mean = torch.tanh(mean)
        return mean


class HyperCategoricalValue(nn.Module):
    """
    A value function that predicts a categorical distribution over a range of values.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_bins: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.w2 = HyperDense(hidden_dim, num_bins, device=device)
        self.bias = nn.Parameter(torch.zeros(num_bins, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.w1(x)
        logits = self.scaler(logits)
        logits = self.w2(logits) + self.bias.to(logits.dtype)
        return logits


class HyperPolicy(nn.Module):
    """
    A policy that outputs a mean
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.mean_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
        self.mean_w2 = HyperDense(hidden_dim, action_dim, device=device)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias
        return mean