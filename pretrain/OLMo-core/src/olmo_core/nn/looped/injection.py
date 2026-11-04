"""Input injections of the looped models, ported from xLLM-Loop ``xllm/models/looped.py``.

Parameter names match the released checkpoints (``input_adapter``, ``A_log``, ``dt_bias``,
``adapter``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.nn.transformer.init import _apply_init, init_linear


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class DiagonalInjection(nn.Module):
    """Diagonal input injection, ``h <- exp(-dt * a) * h + dt * (W e)``.

    ``dt = softplus(dt_bias)`` and ``a = exp(A_log)`` are per channel and ``W`` starts as the
    identity.
    """

    def __init__(self, d_model: int, *, dt_init: float, a_init: float, init_device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if dt_init <= 0 or a_init <= 0:
            raise ValueError("the diagonal injection needs positive dt and a")
        self.dt_init, self.a_init = dt_init, a_init
        self.input_adapter = nn.Linear(d_model, d_model, bias=False, device=init_device, dtype=dtype)
        self.A_log = nn.Parameter(torch.empty(d_model, device=init_device, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(d_model, device=init_device, dtype=dtype))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # The model may already be replicated or sharded (parameters are DTensors then).
            _apply_init(nn.init.eye_, self.input_adapter.weight)
            self.A_log.fill_(math.log(self.a_init))
            self.dt_bias.fill_(_inverse_softplus(self.dt_init))

    def decayed_state_and_update(self, state: torch.Tensor, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``exp(-dt * a) * h`` and ``dt * (W e)``."""
        update = self.input_adapter(inputs)
        if update.dtype != state.dtype:
            raise TypeError(f"input injection dtype mismatch: update {update.dtype}, state {state.dtype}")
        dt = F.softplus(self.dt_bias).to(dtype=state.dtype)
        decay = torch.exp(-dt * torch.exp(self.A_log).to(dtype=state.dtype))
        return state * decay, dt * update

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        retained, injected = self.decayed_state_and_update(state, inputs)
        return retained + injected


class OrthogonalInjection(DiagonalInjection):
    """OrthoInj: the diagonal injection whose decayed state loses its component along the
    injected update."""

    def __init__(self, d_model: int, *, eps: float, **kwargs):
        super().__init__(d_model, **kwargs)
        self.eps = eps

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        retained, injected = self.decayed_state_and_update(state, inputs)
        # Per token, in FP32 under mixed precision; gradients flow through both terms.
        compute_dtype = torch.float64 if retained.dtype == torch.float64 else torch.float32
        value, anchor = retained.to(compute_dtype), injected.to(compute_dtype)
        coefficient = (value * anchor).sum(dim=-1, keepdim=True) / (
            anchor.square().sum(dim=-1, keepdim=True) + self.eps)
        return (value - coefficient * anchor).to(retained.dtype) + injected


class LinearInjection(nn.Module):
    """Huginn-Linear injection ``W [h; e] + b``, from 2d to d."""

    def __init__(self, d_model: int, *, bias: bool, init_std: float, init_device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.init_std = init_std
        self.adapter = nn.Linear(2 * d_model, d_model, bias=bias, device=init_device, dtype=dtype)

    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        init_linear(self.adapter, std=self.init_std, generator=generator)

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        return self.adapter(torch.cat([state, inputs], dim=-1))
