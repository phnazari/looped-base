"""The looped models of *Towards Looped Models Done Right* (Parts I and II) on OLMo-core.

Ported from xLLM-Loop (https://github.com/ifm-ai/xllm-loop, Apache-2.0), ``xllm/models/looped.py``:

- ``looped_transformer`` is its ``LoopedTransformer``: blocks [loop_start_layer, loop_end_layer)
  run loop_times times, optionally re-injecting the embeddings at every loop entry, plus the
  unrolled control that injects an earlier block's output into later blocks.
- ``huginn`` is its ``Huginn``: a prelude, a recurrent block iterated loop_times times from an
  initial state with the prelude output injected at every recurrence, and a coda; the state can be
  split into hierarchical H/L states.
- ``huginn`` with ``depth_control`` is its ``DepthControlledHuginn`` (Part II): the training depth is
  fixed, Poisson-lognormal or drawn from the learned prior, the first draws beyond
  ``backprop_depth`` run without gradients, and the linear, Parcae and orthogonal injections apply.

The blocks are OLMo-core's standard pre-norm blocks, called once per execution; a looped block's
parameters are the same module at every call. Initialization follows the release: truncated normal
with std ``init_std`` for every block matrix, and std ``d_model**-0.5`` for the embeddings and the
output head.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError

from ..layer_norm import RMSNorm
from ..lm_head import LMOutputWithLoss
from ..transformer.init import init_linear
from ..transformer.model import Transformer
from olmo_core.utils import mark_dynamic
from .config import LoopedConfig
from .depth import DepthPriorConfig, LearnedDepthPrior, cumulative_distribution, pln_pmf, sample_depth
from .injection import DiagonalInjection, LinearInjection, OrthogonalInjection

__all__ = ["LoopedTransformer"]


class _LatestKV:
    """Huginn's "latest" cache lookup in a parallel forward (``LoopedConfig.frozen_kv``): every core
    attention keeps each token's keys and values of the last recurrence it ran, and a token marked
    ``frozen`` attends with, and is attended at, those instead of new ones. Stateful across the
    recurrences of one forward, so not for activation checkpointing that recomputes the core."""

    def __init__(self):
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.frozen: Optional[torch.Tensor] = None  # batch x sequence

    def __call__(self, attention: nn.Module, k: torch.Tensor, v: torch.Tensor):
        key = id(attention)
        if key in self.cache and self.frozen is not None:
            cached_k, cached_v = self.cache[key]
            frozen = self.frozen[:, :, None, None]
            k, v = torch.where(frozen, cached_k, k), torch.where(frozen, cached_v, v)
        self.cache[key] = (k, v)
        return k, v


def _loop_schedule(n_layers: int, times: int, start: int, end: int) -> Tuple[int, ...]:
    """The physical-block order of a fixed loop: prefix, the loop span ``times`` times, suffix."""
    return tuple(range(start)) + tuple(range(start, end)) * times + tuple(range(end, n_layers))


class LoopedTransformer(Transformer):
    """A transformer whose blocks run in the order of a looped model (see the module docstring)."""

    def __init__(self, *, looped: LoopedConfig, **kwargs):
        super().__init__(**kwargs)
        looped.validate(self.n_layers)
        self.looped = looped
        d, n = self.d_model, self.n_layers
        init_device = self.embeddings.weight.device.type
        if looped.qk_bias:
            for block in self.blocks.values():
                for linear in (block.attention.w_q, block.attention.w_k):
                    if linear.bias is None:
                        linear.bias = nn.Parameter(
                            torch.zeros(linear.out_features, device=init_device, dtype=linear.weight.dtype))

        def diagonal(dt: float, a: float) -> DiagonalInjection:
            return DiagonalInjection(d, dt_init=dt, a_init=a, init_device=init_device, dtype=self.dtype)

        self.loop_input_injection_module: Optional[DiagonalInjection] = None
        self.dense_prelude_input_injection_module: Optional[DiagonalInjection] = None
        self.input_injection: Optional[nn.Module] = None
        self.huginn_prelude_norm_module: Optional[RMSNorm] = None
        self.huginn_recurrent_exit_norm_module: Optional[RMSNorm] = None

        if looped.architecture == "looped_transformer":
            self.loop_start, self.loop_end = looped.loop_start_layer, looped.end_layer(n)
            self.execution_schedule = _loop_schedule(n, looped.loop_times, self.loop_start, self.loop_end)
            if looped.loop_input_injection != "none":
                self.loop_input_injection_module = diagonal(looped.loop_diagonal_dt_init, looped.loop_diagonal_a_init)
            if looped.prelude_input_injection != "none":
                self.dense_prelude_input_injection_module = diagonal(
                    looped.prelude_diagonal_dt_init, looped.prelude_diagonal_a_init)
            self.prelude_input_layers = frozenset(int(layer) for layer in looped.prelude_input_layers)
        else:
            self.loop_start = looped.prelude_layers
            self.loop_end = self.loop_start + looped.recurrent_count(n)
            if looped.hierarchical_state == "split_hl":
                split = self.loop_start + looped.recurrent_count(n) // 2
                h_layers = tuple(range(self.loop_start, split)) * looped.split_module_repeats
                l_layers = tuple(range(split, self.loop_end)) * looped.split_module_repeats
                cycles = (l_layers * looped.hierarchical_l_cycles + h_layers) * looped.hierarchical_h_cycles
                self.execution_schedule = tuple(range(self.loop_start)) + cycles + tuple(range(self.loop_end, n))
            else:
                self.execution_schedule = _loop_schedule(n, looped.loop_times, self.loop_start, self.loop_end)

            if looped.input_injection == "linear":
                self.input_injection = LinearInjection(
                    d, bias=looped.linear_injection_bias, init_std=self._init_std_value(kwargs),
                    init_device=init_device, dtype=self.dtype)
            elif looped.input_injection == "diagonal":
                if looped.orthogonal_injection:
                    self.input_injection = OrthogonalInjection(
                        d, eps=looped.orthogonal_eps, dt_init=looped.diagonal_dt_init,
                        a_init=looped.diagonal_a_init, init_device=init_device, dtype=self.dtype)
                else:
                    self.input_injection = diagonal(looped.diagonal_dt_init, looped.diagonal_a_init)

            eps = self._norm_eps()
            if looped.prelude_norm:
                self.huginn_prelude_norm_module = RMSNorm(
                    size=d, eps=eps, elementwise_affine=True, bias=False, init_device=init_device)
            if looped.recurrent_exit_norm:
                self.huginn_recurrent_exit_norm_module = RMSNorm(
                    size=d, eps=eps, elementwise_affine=False, bias=False, init_device=init_device)

        # Transformer.__init__ caches the parameter counts before the modules above exist.
        self.__dict__.pop("num_params", None)
        self.__dict__.pop("num_non_embedding_params", None)

        # Training depth (depth control). Every rank draws the same depth from the forward index.
        self.iteration_step = 0
        self.depth_distribution: Tuple[Tuple[int, float], ...] = ((looped.loop_times, 1.0),)
        self.depth_cdf: Optional[Tuple[float, ...]] = None
        self.depth_prior: Optional[LearnedDepthPrior] = None
        if looped.depth_control and looped.sampling == "poisson_lognormal":
            pmf = pln_pmf(looped.poisson_lognormal_mean, looped.poisson_lognormal_sigma,
                          looped.poisson_lognormal_max)
            self.depth_distribution = tuple(enumerate(pmf, start=1))
            self.depth_cdf = cumulative_distribution(pmf)
            if looped.learned_depth_prior:
                self.depth_prior = LearnedDepthPrior(
                    DepthPriorConfig(entropy_coefficient=looped.depth_prior_entropy,
                                     target_mean=looped.poisson_lognormal_mean), initial_pmf=pmf)
        # The state at the last optimizer step, restored after a dry-run batch (see post_batch).
        self._step_start = self._depth_state()
        self.depth_prior_metrics: Dict[str, float] = {}

        self.exit_head: Optional[nn.Linear] = None
        if looped.exit_head:
            self.exit_head = nn.Linear(d, 1, bias=True, device=init_device, dtype=self.dtype)
        self._exit_logits: Optional[torch.Tensor] = None  # recurrences x batch x sequence, last forward
        self.last_exit_counts: Optional[torch.Tensor] = None  # recurrences run, last forward with exit_threshold
        self._exit_stats: Optional[torch.Tensor] = None  # BCE sum, exact exits, supervised tokens

    @staticmethod
    def _init_std_value(kwargs: Dict[str, Any]) -> float:
        return float(kwargs.get("init_std", 0.02))

    def _norm_eps(self) -> float:
        first = next(iter(self.blocks.values()))
        return float(getattr(first.attention_norm, "eps", 1e-6))

    # ---------------------------------------------------------------------------------------------
    # Initialization

    @torch.no_grad()
    def init_weights(self, *args, **kwargs) -> torch.Generator:
        generator = super().init_weights(*args, **kwargs)
        # The release initializes the output head like the embeddings, std 1/sqrt(d).
        if self.lm_head is not None:
            init_linear(self.lm_head.w_out, std=self.d_model**-0.5, generator=generator)
        # The base initialization resets each injection before its inner linear, which then
        # overwrites the identity input adapter; initialize the injections again, last.
        for injection in (self.input_injection, self.loop_input_injection_module,
                          self.dense_prelude_input_injection_module):
            if isinstance(injection, LinearInjection):
                injection.reset_parameters(generator=generator)
            elif isinstance(injection, DiagonalInjection):
                injection.reset_parameters()
        # The exit head starts silent: zero weights, a low bias.
        if self.exit_head is not None:
            nn.init.zeros_(self.exit_head.weight)
            nn.init.constant_(self.exit_head.bias, self.looped.exit_bias_init)
        return generator

    # ---------------------------------------------------------------------------------------------
    # Forward

    def _run_blocks(self, layer_ids: Sequence[int], h: torch.Tensor, all_block_kwargs: Dict[str, Any],
                    per_block_kwargs: Dict[int, Dict[str, Any]]) -> torch.Tensor:
        for layer_id in layer_ids:
            if self.compile_enabled:
                mark_dynamic(h, (0, 1), strict=False)
            h = self.blocks[str(layer_id)](h, **all_block_kwargs, **per_block_kwargs.get(layer_id, {}))
        return h

    def initialize_recurrent_state(self, input_embeds: torch.Tensor, cache_len: int = 0,
                                   seeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """With ``seeds`` (one per sequence) the truncated normal is drawn by the release's position hash
        (``_hashed_normal_state``), clamped at 3 std: it depends only on the seed, the position and the hidden
        index, so two forwards of a sequence start from the same state whatever their length or batch."""
        init = self.looped.state_init
        if seeds is not None:
            if init != "truncated_normal":
                raise OLMoConfigurationError("state seeds need state_init truncated_normal")
            std = self.looped.state_init_std
            term = (seeds % 99991).float().view(-1, 1, 1) * 0.037719
            state = self._hashed_normal_state(input_embeds.float(), cache_len, term, 1.0).clamp_(-3, 3).mul_(std)
            state = state.to(input_embeds.dtype)
            return state if self.embed_scale is None else state * self.embed_scale
        if init == "zero":
            return torch.zeros_like(input_embeds)
        if init == "input":
            return input_embeds
        if init == "truncated_normal":
            std = self.looped.state_init_std
            state = nn.init.trunc_normal_(torch.empty_like(input_embeds), std=std, a=-3 * std, b=3 * std)
            return state if self.embed_scale is None else state * self.embed_scale
        if not self.training:
            batch = torch.arange(input_embeds.shape[0], device=input_embeds.device, dtype=torch.float32)
            return self._hashed_normal_state(
                input_embeds, cache_len, batch.view(-1, 1, 1) * 0.037719, math.sqrt(2.0 / self.d_model))
        return torch.randn_like(input_embeds).mul(self.d_model**-0.5)

    @staticmethod
    def _hashed_normal_state(input_embeds: torch.Tensor, cache_len: int, batch_term: torch.Tensor,
                             scale: float) -> torch.Tensor:
        """The release's deterministic evaluation stand-in for the training state N(0, 1/d): each
        entry hashes its hidden index, position and a per-row term, fract(43758.5453123 * sin(x))
        with x = 0.013579 * dim + 0.021713 * pos + batch_term + 0.123457, mapped to a normal by
        sqrt(2) * erfinv(2u - 1), times ``scale``."""
        batch_size, seq_len, hidden_dim = input_embeds.shape
        device = input_embeds.device
        dimension = torch.arange(hidden_dim, device=device, dtype=torch.float32).view(1, 1, -1)
        position = torch.arange(cache_len, cache_len + seq_len, device=device, dtype=torch.float32).view(1, -1, 1)
        state = dimension.expand(batch_size, seq_len, hidden_dim).clone()
        state.mul_(0.013579)
        state.add_(position * 0.021713)
        state.add_(batch_term)
        state.add_(0.123457)
        state.sin_()
        state.mul_(43758.5453123)
        state.sub_(state.floor())
        state.clamp_(1e-6, 1.0 - 1e-6)
        state.mul_(2.0).sub_(1.0)
        state.erfinv_()
        state.mul_(scale)
        return state.to(dtype=input_embeds.dtype)

    def _recurrence_input(self, prelude_output: torch.Tensor) -> torch.Tensor:
        if self.huginn_prelude_norm_module is None:
            return prelude_output
        return self.huginn_prelude_norm_module(prelude_output)

    def _recur(self, state, input_embeds, start: int, end: int, kwargs, module_repeats: int = 1):
        """One state update: the input injection, then blocks [start, end) module_repeats times, then
        the exit norm if any."""
        if self.input_injection is not None:
            state = self.input_injection(state, input_embeds)
        for _ in range(module_repeats):
            state = self._run_blocks(range(start, end), state, *kwargs)
        if self.huginn_recurrent_exit_norm_module is not None:
            state = self.huginn_recurrent_exit_norm_module(state)
        return state

    def _draw_recurrence_steps(self) -> Tuple[int, int]:
        """The recurrences to run without and with gradients; a training forward advances the draws."""
        depth = self.looped.loop_times
        if self.training and self.looped.depth_control:
            if self.depth_prior is not None:
                depth = self.depth_prior.sample()
            elif self.depth_cdf is not None:
                depth = sample_depth(self.depth_cdf, self.iteration_step, antithetic=self.looped.antithetic_sampling)
            self.iteration_step += 1
        backprop = self.looped.backprop_depth
        with_grad = depth if backprop is None else min(depth, backprop)
        return depth - with_grad, with_grad

    def _run_recurrences(self, state, input_embeds, kwargs, loop_counts: Optional[torch.Tensor] = None,
                         collect_exits: bool = False, exit_mask: Optional[torch.Tensor] = None,
                         exit_threshold: Optional[float] = None):
        """With ``loop_counts`` (batch x sequence) token t's state is frozen after ``loop_counts[t]``
        recurrences; later recurrences still read it (keys and values of the frozen state), as with
        the per-token exits of the Huginn release. The drawn depth must cover the largest count.

        With ``collect_exits`` the exit head's logits after every recurrence are kept; with
        ``exit_threshold`` a token of ``exit_mask`` is frozen after the first recurrence whose exit
        probability exceeds it (``last_exit_counts``)."""
        looped = self.looped
        start, end = self.loop_start, self.loop_end
        if looped.hierarchical_state == "none":
            without_grad, with_grad = self._draw_recurrence_steps()
            depth = without_grad + with_grad
            if loop_counts is None and not collect_exits:
                with torch.no_grad():
                    for _ in range(without_grad):
                        state = self._recur(state, input_embeds, start, end, kwargs)
                for _ in range(with_grad):
                    state = self._recur(state, input_embeds, start, end, kwargs)
                return state
            if loop_counts is None:
                loop_counts = torch.full(state.shape[:2], depth, device=state.device, dtype=torch.long)
            elif int(loop_counts.max()) > depth or int(loop_counts.min()) < 1:
                raise ValueError("loop_counts must lie in [1, depth]")
            halting = exit_threshold is not None
            if halting:
                loop_counts = loop_counts.clone()
                running = exit_mask.bool() & (loop_counts == depth)
            latest = None
            if looped.frozen_kv:
                latest = _LatestKV()
                kwargs = ({**kwargs[0], "kv_hook": latest}, kwargs[1])
            logits = []
            for step in range(depth):
                if latest is not None:
                    latest.frozen = step >= loop_counts
                with torch.no_grad() if step < without_grad else contextlib.nullcontext():
                    previous = state
                    update = self._recur(state, input_embeds, start, end, kwargs)
                    state = torch.where((step < loop_counts).unsqueeze(-1), update, state)
                    if collect_exits:
                        step_logits = self._exit_logits_at(state, previous)
                        logits.append(step_logits)
                        if halting:
                            halt = running & (step_logits.sigmoid() > exit_threshold)
                            loop_counts = torch.where(halt, step + 1, loop_counts)
                            running = running & ~halt
            if collect_exits:
                self._exit_logits = torch.stack(logits)
            if halting:
                self.last_exit_counts = loop_counts
            return state
        if loop_counts is not None:
            raise ValueError("loop_counts supports a single recurrent state")
        # The caller's state is not used: z_H starts from the prelude output and z_L from a second
        # draw, as in the release.
        z_h = input_embeds
        z_l = self.initialize_recurrent_state(input_embeds)
        if looped.hierarchical_state == "split_hl":
            split = start + looped.recurrent_count(self.n_layers) // 2
            h_span, l_span = (start, split), (split, end)
        else:
            h_span = l_span = (start, end)
        for _ in range(looped.hierarchical_h_cycles):
            for _ in range(looped.hierarchical_l_cycles):
                z_l = self._recur(z_l + z_h, input_embeds, *l_span, kwargs, looped.split_module_repeats)
            z_h = self._recur(z_h + z_l, input_embeds, *h_span, kwargs, looped.split_module_repeats)
        return z_h

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
        return_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        loop_counts: Optional[torch.Tensor] = None,
        exit_targets: Optional[torch.Tensor] = None,
        exit_weights: Optional[torch.Tensor] = None,
        exit_mask: Optional[torch.Tensor] = None,
        exit_threshold: Optional[float] = None,
        state_seeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, LMOutputWithLoss]:
        """``exit_targets`` (batch x sequence, 0 for tokens without one) give the recurrence each token
        should exit at, and ``exit_weights`` the weight of its exit loss (see ``_exit_loss``); the
        weighted sum is added to the loss. ``exit_mask`` and
        ``exit_threshold`` halt tokens by the exit head (see ``_run_recurrences``); ``state_seeds`` (one per
        sequence) make the recurrent state's start deterministic (see ``initialize_recurrent_state``)."""
        if loop_counts is not None and self.looped.architecture != "huginn":
            raise OLMoConfigurationError("loop_counts needs a huginn model")
        collect_exits = exit_targets is not None or exit_threshold is not None
        if collect_exits and self.looped.exit_rule == "head" and self.exit_head is None:
            raise OLMoConfigurationError("exit targets and halting by the head need looped.exit_head")
        if exit_threshold is not None and exit_mask is None:
            raise ValueError("halting by the exit head needs exit_mask")
        input_ids, labels, all_block_kwargs, per_block_kwargs, lm_head_kwargs = self._prepare_inputs(
            input_ids, labels, ignore_index=ignore_index, loss_reduction=loss_reduction,
            z_loss_multiplier=z_loss_multiplier, loss_div_factor=loss_div_factor,
            return_logits=return_logits, logits_to_keep=logits_to_keep, **kwargs,
        )
        block_kwargs = (all_block_kwargs, per_block_kwargs)
        h = self.embeddings(input_ids)
        if self.embed_scale is not None:
            h = h * self.embed_scale
        if self.embedding_norm is not None:
            h = self.embedding_norm(h)

        if self.looped.architecture == "looped_transformer":
            embeddings, source = h, None
            for layer_id in self.execution_schedule:
                if self.loop_input_injection_module is not None and layer_id == self.loop_start:
                    h = self.loop_input_injection_module(h, embeddings)
                if self.dense_prelude_input_injection_module is not None and layer_id in self.prelude_input_layers:
                    h = self.dense_prelude_input_injection_module(h, source)
                h = self._run_blocks((layer_id,), h, *block_kwargs)
                if self.dense_prelude_input_injection_module is not None and layer_id == self.looped.prelude_source_layer:
                    source = h
        else:
            h = self._run_blocks(range(self.loop_start), h, *block_kwargs)
            input_embeds = self._recurrence_input(h)
            h = self.initialize_recurrent_state(input_embeds, seeds=state_seeds)
            h = self._run_recurrences(h, input_embeds, block_kwargs, loop_counts, collect_exits=collect_exits,
                                      exit_mask=exit_mask, exit_threshold=exit_threshold)
            if self.looped.recurrence_final_norm:
                h = self.lm_head.norm(h)
            h = self._run_blocks(range(self.loop_end, self.n_layers), h, *block_kwargs)

        if self.compile_enabled:
            mark_dynamic(h, (0, 1), strict=False)
            if labels is not None:
                mark_dynamic(labels, (0, 1), strict=False)
        if labels is not None:
            lm_head_kwargs["labels"] = labels
        output = self.lm_head(h, **lm_head_kwargs)
        if self.training and self.depth_prior is not None and labels is not None:
            self._observe(output, labels, ignore_index, loss_reduction, loss_div_factor)
        if exit_targets is not None:
            exit_loss = self._exit_loss(exit_targets, exit_weights)
            if isinstance(output, LMOutputWithLoss):
                output = output._replace(loss=output.loss + self.looped.exit_loss_weight * exit_loss)
        return output

    def _exit_logits_at(self, state: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Exit logits after a recurrence (batch x sequence): the exit head's, or Huginn's latent-diff rule
        (``LoopedConfig.exit_rule``). A frozen token's change is zero, so the rule fires on it."""
        looped = self.looped
        if looped.exit_rule == "head":
            return self.exit_head(self.lm_head.norm(state)).squeeze(-1).float()
        change = (state - previous).float().norm(dim=-1) / state.float().norm(dim=-1).clamp_min(1e-6)
        return (math.log(looped.exit_latent_threshold) - change.clamp_min(1e-8).log()) / looped.exit_latent_temperature

    def _exit_loss(self, targets: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        """The weighted exit loss of the last forward, and its metrics: the mean BCE and the rate of
        tokens whose head first fires exactly at the target. The head is a hazard: per token the mean
        BCE over recurrences 1..target of not firing before the target and firing at it. Later
        recurrences are not supervised (a token frozen at its target repeats its state there)."""
        logits = self._exit_logits
        steps = torch.arange(1, logits.shape[0] + 1, device=logits.device).view(-1, 1, 1)
        mask = targets > 0
        supervised = (steps <= targets.unsqueeze(0)) & mask
        bce = F.binary_cross_entropy_with_logits(
            logits, (steps == targets.unsqueeze(0)).float(), reduction="none")
        bce = (bce * supervised).sum(0) / supervised.sum(0).clamp(min=1)
        if weights is None:
            weights = mask.float() / mask.sum().clamp(min=1)
        loss = (bce * weights * mask).sum()
        fired = logits > 0
        first = torch.where(fired.any(0), fired.float().argmax(0) + 1, logits.shape[0] + 1)
        stats = torch.stack([(bce * mask).sum().detach(), ((first == targets) & mask).sum().float(),
                             mask.sum().float()])
        self._exit_stats = stats if self._exit_stats is None else self._exit_stats + stats
        return loss

    def compute_auxiliary_metrics(self, reset: bool = True):
        from olmo_core.train.common import ReduceType

        out = super().compute_auxiliary_metrics(reset=reset)
        if self._exit_stats is not None:
            bce, exact, count = self._exit_stats.unbind()
            out["exit/bce"] = (bce / count.clamp(min=1), ReduceType.mean)
            out["exit/exact"] = (exact / count.clamp(min=1), ReduceType.mean)
            if reset:
                self._exit_stats = None
        return out

    def reset_auxiliary_metrics(self):
        super().reset_auxiliary_metrics()
        self._exit_stats = None

    # ---------------------------------------------------------------------------------------------
    # Learned depth prior and the training state of the depth draws

    def _observe(self, output, labels, ignore_index, loss_reduction, loss_div_factor) -> None:
        """Hand the prior this forward's token losses, as the release's ``after_forward`` hook does:
        their sum and count, which the prior reduces over ranks."""
        if not isinstance(output, LMOutputWithLoss) or loss_reduction == "none":
            raise RuntimeError("the learned depth prior needs a reduced CE loss from the forward")
        mask = labels != ignore_index
        count = mask.sum()
        ce = output.ce_loss.detach().double()
        if loss_reduction == "mean":
            total = ce * count
        else:
            total = ce * (loss_div_factor if loss_div_factor is not None else 1.0)
        assert self.depth_prior is not None
        # The prior sums the losses and the mask; the summed loss and the token count give it the
        # same mean as the per-token losses would.
        self.depth_prior.observe(total.reshape(1), count.reshape(1).double())

    def _depth_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"iteration_step": self.iteration_step}
        if self.depth_prior is not None:
            state["depth_prior_rng"] = self.depth_prior.rng.get_state()
        return state

    def post_batch(self, dry_run: bool = False):
        """A dry-run batch must not advance the depth draws: restore the state of the last step."""
        super().post_batch(dry_run=dry_run)
        if not dry_run:
            return
        self.iteration_step = self._step_start["iteration_step"]
        if self.depth_prior is not None:
            self.depth_prior.rng.set_state(self._step_start["depth_prior_rng"])
            self.depth_prior.pending_depths.clear()
            self.depth_prior.pending_losses.clear()
            self.depth_prior.pending_counts.clear()

    def post_optim_step(self):
        """After every optimizer step: the prior's REINFORCE update, as the release's
        ``after_optimizer_step`` hook."""
        super().post_optim_step()
        if self.depth_prior is not None:
            self.depth_prior_metrics = {
                f"depth_prior/{key}": value for key, value in self.depth_prior.finish_step().items()}
        self._step_start = self._depth_state()

    def training_state_dict(self) -> Dict[str, Any]:
        """Training state the weights do not hold: the forward index and the prior."""
        state: Dict[str, Any] = {"iteration_step": self.iteration_step}
        if self.depth_prior is not None:
            state["depth_prior"] = self.depth_prior.state_dict()
        return state

    def load_training_state_dict(self, state: Dict[str, Any]) -> None:
        self.iteration_step = int(state["iteration_step"])
        if self.depth_prior is not None:
            self.depth_prior.load_state_dict(state["depth_prior"])
        self._step_start = self._depth_state()

    # ---------------------------------------------------------------------------------------------
    # Accounting

    def apply_pp(self, pp_mesh):
        raise OLMoConfigurationError("looped models do not support pipeline parallelism")

    def apply_tp(self, *args, **kwargs):
        raise OLMoConfigurationError("looped models do not support tensor parallelism")

    def apply_cp(self, *args, **kwargs):
        raise OLMoConfigurationError("looped models do not support context parallelism")

    def num_flops_per_token(self, seq_len: int) -> int:
        """Training FLOPs per token in OLMo-core's convention (each block's
        ``num_flops_per_token`` per call), counting every block call of the execution and, with
        depth control, the expected depth: recurrences outside the backward window count only
        their forward (a third)."""
        block_flops = {int(key): block.num_flops_per_token(seq_len) for key, block in self.blocks.items()}
        injection_flops = 6 * self.looped.num_params(self.d_model)
        total = self.lm_head.num_flops_per_token(seq_len) if self.lm_head is not None else 0
        if self.looped.architecture == "looped_transformer" or self.looped.hierarchical_state != "none":
            total += sum(block_flops[layer] for layer in self.execution_schedule)
            if self.looped.architecture == "huginn" and self.input_injection is not None:
                looped = self.looped
                total += looped.hierarchical_h_cycles * (looped.hierarchical_l_cycles + 1) * injection_flops
            elif self.looped.architecture == "looped_transformer":
                calls = self.looped.loop_times * (self.loop_input_injection_module is not None) + sum(
                    layer in self.prelude_input_layers for layer in self.execution_schedule
                ) * (self.dense_prelude_input_injection_module is not None)
                total += calls * 6 * (self.d_model * self.d_model + 2 * self.d_model)
            return int(total)
        static = sum(block_flops[layer] for layer in range(self.loop_start)) + sum(
            block_flops[layer] for layer in range(self.loop_end, self.n_layers))
        recurrence = sum(block_flops[layer] for layer in range(self.loop_start, self.loop_end)) + injection_flops
        backprop = self.looped.backprop_depth
        expected = 0.0
        for depth, probability in self.depth_distribution:
            with_grad = depth if backprop is None else min(depth, backprop)
            expected += probability * (with_grad + (depth - with_grad) / 3.0)
        return int(total + static + expected * recurrence)

