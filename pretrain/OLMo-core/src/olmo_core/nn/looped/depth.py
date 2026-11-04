"""Recurrent-depth distributions, draws and the learned depth prior of *Towards Looped Models Done
Right, Part II*.

Copied from xLLM-Loop (https://github.com/ifm-ai/xllm-loop, Apache-2.0),
``xllm/models/looped_depth.py`` and ``xllm/models/looped_depth_prior.py``, with only the imports
merged: the arithmetic is unchanged, so given the same forward index (and, for the learned prior,
the same per-forward losses) the draws and the prior's steps equal the paper runs'.

Fixed PLN-5 samples each training forward's depth from min(cap, 1 + Poisson(λ)) with
log λ ~ N(μ, σ²), where μ is calibrated so that the capped mean equals the target.

The learned prior is a categorical distribution over depths 1..64, stored as float64 logits on the
CPU, outside the model and its optimizer. Every training forward draws one depth from it; after each
optimizer step it takes a REINFORCE step toward depths whose inverse perplexity exp(-CE) beats an
EMA baseline. The advantage is rescaled into cross-entropy units by the ratio of the EMA standard
deviations of CE and of the cost; a quadratic penalty holds the mean depth at the target, and an
entropy term (lambda_H) keeps the distribution spread.
"""

from __future__ import annotations

import copy
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Sequence

import torch
import torch.distributed as dist


PLN_SEED_OFFSET = 514229

# The PLN-5 PMF (target mean 5, sigma 0.5, cap 64) the paper's training runs used,
# as computed in their environment with torch 2.8.0+cu128. Other CPUs and torch
# builds change `calibrated_capped_poisson_lognormal` in the last bits.
PAPER_PLN5_PMF = (
    0.05768747546482591,
    0.1271367387177668,
    0.1634086392890813,
    0.1619074983340164,
    0.13796282734225382,
    0.10698272319110169,
    0.07804764707860895,
    0.05470693907876852,
    0.037361700976777726,
    0.025099433949814245,
    0.016697717702268286,
    0.011052629620047851,
    0.007304055652903804,
    0.004830737772361538,
    0.003203151771039994,
    0.002132081017445029,
    0.0014258721987138567,
    0.0009586921076838706,
    0.0006483099746274615,
    0.0004410716241755722,
    0.000301944551073642,
    0.00020800269392146997,
    0.00014419125207641085,
    0.00010058218331207551,
    7.059667258419262e-05,
    4.9852412062464184e-05,
    3.5414136427653425e-05,
    2.5304622107571485e-05,
    1.8184285152127846e-05,
    1.3140259666362847e-05,
    9.546846563012203e-06,
    6.972695919262932e-06,
    5.118738343708846e-06,
    3.7764502416124336e-06,
    2.7996353510214844e-06,
    2.085238462098128e-06,
    1.5602277964955316e-06,
    1.1725776685897305e-06,
    8.850369680325107e-07,
    6.708005361757733e-07,
    5.104868279453078e-07,
    3.900165388933937e-07,
    2.9911563900765575e-07,
    2.3025345865338855e-07,
    1.778857064050211e-07,
    1.3791253597731962e-07,
    1.0728910765229482e-07,
    8.3744719506e-08,
    6.557941898380174e-08,
    5.151599353655518e-08,
    4.059163481228815e-08,
    3.207815872458309e-08,
    2.5422939995903076e-08,
    2.020502320279681e-08,
    1.6102459981160398e-08,
    1.286800205515741e-08,
    1.0311018925701726e-08,
    8.284031836490163e-09,
    6.672645135115369e-09,
    5.387960542342231e-09,
    4.3608027065798596e-09,
    3.5372756494056925e-09,
    2.8753142244955275e-09,
    1.3322268843651841e-08,
)


@lru_cache(maxsize=None)
def standard_normal_gauss_hermite(order: int = 64) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Gauss-Hermite nodes and weights for expectations under N(0, 1)."""
    jacobi = torch.zeros((order, order), dtype=torch.float64, device="cpu")
    off_diagonal = torch.sqrt(torch.arange(1, order, dtype=torch.float64, device="cpu") / 2.0)
    jacobi.diagonal(1).copy_(off_diagonal)
    jacobi.diagonal(-1).copy_(off_diagonal)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    normal_nodes = math.sqrt(2.0) * nodes
    normal_weights = eigenvectors[0].square()
    return tuple(normal_nodes.tolist()), tuple(normal_weights.tolist())


def capped_poisson_lognormal_pmf(mu: float, sigma: float, max_steps: int) -> tuple[float, ...]:
    """Return the PMF of min(max_steps, 1 + Poisson(LogNormal(mu, sigma)))."""
    nodes, weights = standard_normal_gauss_hermite()
    nodes_tensor = torch.tensor(nodes, dtype=torch.float64, device="cpu")
    weights_tensor = torch.tensor(weights, dtype=torch.float64, device="cpu")
    rates = torch.exp(mu + sigma * nodes_tensor)
    log_rates = torch.log(rates)  # not mu + sigma * nodes: the paper code's rounding
    probabilities = []
    for poisson_count in range(max_steps - 1):
        log_probability = -rates + poisson_count * log_rates - math.lgamma(poisson_count + 1)
        probabilities.append(float(torch.sum(weights_tensor * torch.exp(log_probability)).item()))
    probabilities.append(max(0.0, 1.0 - sum(probabilities)))
    normalization = sum(probabilities)
    return tuple(probability / normalization for probability in probabilities)


@lru_cache(maxsize=None)
def calibrated_capped_poisson_lognormal(
    target_mean: float, sigma: float, max_steps: int
) -> tuple[float, tuple[float, ...]]:
    """Bisect μ so the capped depth mean equals `target_mean`; return (μ, PMF).

    [-20, 20] brackets every target mean between 1 and the cap, and 80 halvings reach float64 resolution.
    """
    lower_mu, upper_mu = -20.0, 20.0
    for _ in range(80):
        mu = (lower_mu + upper_mu) / 2.0
        pmf = capped_poisson_lognormal_pmf(mu, sigma, max_steps)
        mean = sum(step * probability for step, probability in enumerate(pmf, start=1))
        if mean < target_mean:
            lower_mu = mu
        else:
            upper_mu = mu
    calibrated_mu = (lower_mu + upper_mu) / 2.0
    return calibrated_mu, capped_poisson_lognormal_pmf(calibrated_mu, sigma, max_steps)


def pln_pmf(target_mean: float, sigma: float, max_steps: int) -> tuple[float, ...]:
    """Return the capped PLN PMF, using the paper runs' exact values for PLN-5."""
    if (target_mean, sigma, max_steps) == (5.0, 0.5, 64):
        return PAPER_PLN5_PMF
    return calibrated_capped_poisson_lognormal(target_mean, sigma, max_steps)[1]


def cumulative_distribution(pmf: tuple[float, ...]) -> tuple[float, ...]:
    """Return the CDF used for inverse-transform draws; its last entry is 1."""
    cumulative_probability = 0.0
    cdf = []
    for probability in pmf:
        cumulative_probability += probability
        cdf.append(min(cumulative_probability, 1.0))
    cdf[-1] = 1.0
    return tuple(cdf)


def sample_depth(
    cdf: tuple[float, ...],
    iteration_step: int,
    *,
    antithetic: bool,
    seed_offset: int = PLN_SEED_OFFSET,
) -> int:
    """Draw one depth for the given training forward index.

    Every rank draws the same depth from a CPU generator seeded by the forward
    index. Antithetic draws pair consecutive forwards as u and 1 - u.
    """
    sampling_step = iteration_step // 2 if antithetic else iteration_step
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling_step + seed_offset)
    quantile = float(torch.rand((), generator=generator).item())
    if antithetic and iteration_step % 2 == 1:
        quantile = 1.0 - quantile
    return min(bisect_left(cdf, quantile) + 1, len(cdf))

MAX_DEPTH = 64
SUPPORT = torch.arange(1, MAX_DEPTH + 1, dtype=torch.float64)


@dataclass(frozen=True)
class DepthPriorConfig:
    """Hyperparameters of the learned prior (defaults are the paper's)."""

    entropy_coefficient: float
    lr: float = 1e-3
    mean_penalty: float = 1.0
    target_mean: float = 5.0
    baseline_decay: float = 0.95
    burnin_steps: int = 32
    seed: int = PLN_SEED_OFFSET
    std_floor: float = 1e-12


class LearnedDepthPrior:
    """Categorical depth prior trained alongside the model (not an nn.Module).

    `baseline` and `cost_variance` are EMAs of the weighted mean and variance of the
    cost -exp(-CE) over optimizer steps; `raw_baseline` and `raw_variance` track the
    same for CE itself. These names are part of the saved state and the logged metrics.
    """

    def __init__(self, config: DepthPriorConfig, initial_pmf: Sequence[float]) -> None:
        p = torch.as_tensor(initial_pmf, dtype=torch.float64, device="cpu")
        if p.shape != (MAX_DEPTH,) or not torch.isfinite(p).all() or not (p > 0).all():
            raise ValueError(f"initial PMF must be a positive distribution over depths 1..{MAX_DEPTH}")
        if abs(p.sum().item() - 1) >= 1e-10 or abs(p.dot(SUPPORT).item() - config.target_mean) >= 1e-10:
            raise ValueError("initial PMF must sum to 1 and have the target mean")
        self.config = config
        self.logits = torch.nn.Parameter(p.log())
        self.optimizer = torch.optim.Adam([self.logits], lr=config.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.rng = torch.Generator().manual_seed(config.seed)
        self.baseline = 0.0
        self.raw_baseline = 0.0
        self.raw_variance = 0.0
        self.cost_variance = 0.0
        self.updates = 0
        self.pending_depths: list[int] = []
        self.pending_losses: list[float] = []
        self.pending_counts: list[float] = []

    def sample(self) -> int:
        """Draw the depth of the next training forward (identical on every rank)."""
        if len(self.pending_depths) != len(self.pending_losses):
            raise RuntimeError("the previous forward was not observed")
        depth = int(torch.multinomial(self.probabilities(), 1, generator=self.rng).item()) + 1
        self.pending_depths.append(depth)
        return depth

    def observe(self, token_losses: torch.Tensor, token_mask: torch.Tensor | None = None) -> None:
        """Record the global mean token loss of the forward that used the last draw."""
        if len(self.pending_depths) != len(self.pending_losses) + 1:
            raise RuntimeError("observe() must follow sample()")
        total = token_losses.detach().double().sum()
        count = (
            torch.tensor(token_losses.numel(), device=total.device, dtype=torch.float64)
            if token_mask is None
            else token_mask.detach().double().sum()
        )
        pair = torch.stack((total, count))
        if dist.is_initialized():
            dist.all_reduce(pair)
        if not torch.isfinite(pair).all() or pair[1] <= 0:
            raise ValueError("nonfinite loss or empty forward")
        self.pending_losses.append(float((pair[0] / pair[1]).cpu()))
        self.pending_counts.append(float(pair[1].cpu()))

    def finish_step(self) -> dict[str, float]:
        """Update the prior once per optimizer step and return its metrics."""
        if not self.pending_depths or len(self.pending_depths) != len(self.pending_losses):
            raise RuntimeError("finish_step() needs every drawn forward observed")
        raw = torch.tensor(self.pending_losses, dtype=torch.float64)
        costs = -(-raw).exp()
        counts = torch.tensor(self.pending_counts, dtype=torch.float64)
        weights = counts / counts.sum()
        scale = self._advantage_scale()
        loss = self._loss(costs, weights, scale)
        if self.updates >= self.config.burnin_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(self.logits.grad).all():
                raise FloatingPointError("nonfinite depth-prior gradient")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        raw_observed, observed = self._update_baselines(raw, costs, weights)
        self.updates += 1

        metrics = self.metrics()
        metrics.update(
            signal_scale=scale,
            observed_raw_ce=raw_observed,
            observed_cost=observed,
            sampled_step_mean=sum(self.pending_depths) / len(self.pending_depths),
        )
        self.pending_depths.clear()
        self.pending_losses.clear()
        self.pending_counts.clear()
        after = self.probabilities()
        if not torch.isfinite(after).all() or not (after > 0).all():
            raise FloatingPointError("the depth prior lost full support")
        return metrics

    def _advantage_scale(self) -> float:
        """Rescale the cost advantage into CE units: EMA std of CE over EMA std of the cost."""
        if self.updates == 0:
            return 1.0
        floor = self.config.std_floor
        return max(math.sqrt(self.raw_variance), floor) / max(math.sqrt(self.cost_variance), floor)

    def _loss(self, costs: torch.Tensor, weights: torch.Tensor, scale: float) -> torch.Tensor:
        """REINFORCE term plus the mean-depth penalty and the entropy term."""
        config = self.config
        logp = self.logits.log_softmax(0)
        p = logp.exp()
        reinforce_term = (((costs - self.baseline) * scale).detach() * weights
                          * logp[torch.tensor(self.pending_depths) - 1]).sum()
        mean_penalty_term = config.mean_penalty * (p.dot(SUPPORT) - config.target_mean).square()
        entropy_term = config.entropy_coefficient * (p * logp).sum()
        return sum((reinforce_term, mean_penalty_term, entropy_term))

    def _update_baselines(self, raw: torch.Tensor, costs: torch.Tensor, weights: torch.Tensor) -> tuple[float, float]:
        """Fold this step's weighted CE and cost statistics into the EMAs; return the step's means."""
        raw_observed = float((raw * weights).sum())
        observed = float((costs * weights).sum())
        raw_within = float((weights * (raw - raw_observed).square()).sum())
        cost_within = float((weights * (costs - observed).square()).sum())
        decay = self.config.baseline_decay
        if self.updates == 0:
            self.raw_variance, self.cost_variance = raw_within, cost_within
            self.raw_baseline, self.baseline = raw_observed, observed
        else:
            self.raw_variance = (decay * self.raw_variance + (1 - decay) * raw_within
                                 + decay * (1 - decay) * (raw_observed - self.raw_baseline) ** 2)
            self.cost_variance = (decay * self.cost_variance + (1 - decay) * cost_within
                                  + decay * (1 - decay) * (observed - self.baseline) ** 2)
            self.raw_baseline = decay * self.raw_baseline + (1 - decay) * raw_observed
            self.baseline = decay * self.baseline + (1 - decay) * observed
        return raw_observed, observed

    def probabilities(self) -> torch.Tensor:
        return self.logits.detach().softmax(0)

    def metrics(self) -> dict[str, float]:
        p = self.probabilities()
        result = {f"p_{i + 1:02d}": float(v) for i, v in enumerate(p)}
        result.update(
            mean=float(p.dot(SUPPORT)),
            entropy=float(-(p * p.log()).sum()),
            baseline=self.baseline,
            raw_ce_baseline=self.raw_baseline,
            raw_ce_variance=self.raw_variance,
            cost_variance=self.cost_variance,
            updates=self.updates,
        )
        return result

    _STATE_KEYS = ("baseline", "raw_baseline", "raw_variance", "cost_variance", "updates")

    def state_dict(self) -> dict:
        if self.pending_depths or self.pending_losses:
            raise RuntimeError("checkpoint the prior only at an optimizer-step boundary")
        return copy.deepcopy(dict(
            version=1, config=asdict(self.config), logits=self.logits.detach(),
            optimizer=self.optimizer.state_dict(), rng=self.rng.get_state(),
            **{key: getattr(self, key) for key in self._STATE_KEYS},
        ))

    def load_state_dict(self, state: dict) -> None:
        if state["version"] != 1 or state["config"] != asdict(self.config):
            raise ValueError("depth-prior state does not match this configuration")
        with torch.no_grad():
            self.logits.copy_(state["logits"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["rng"])
        for key in self._STATE_KEYS:
            setattr(self, key, copy.deepcopy(state[key]))
