"""Tests of the looped-model port (olmo_core.nn.looped) against the release's definitions.

Run from the repository root:
    PYTHONPATH=pretrain/OLMo-core/src:. python -m pytest -q looped/tests
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.looped import LoopedConfig
from olmo_core.nn.looped.depth import (
    PAPER_PLN5_PMF,
    calibrated_capped_poisson_lognormal,
    cumulative_distribution,
    sample_depth,
)
from olmo_core.nn.looped.injection import DiagonalInjection, LinearInjection, OrthogonalInjection
from olmo_core.nn.looped.model import LoopedTransformer
from olmo_core.nn.rope import RoPEType
from olmo_core.nn.transformer import TransformerConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "looped" / "configs"
D, SEQ = 32, 16


def _model(n_layers: int, **looped) -> LoopedTransformer:
    config = TransformerConfig.llama_like(
        d_model=D, n_layers=n_layers, n_heads=4, n_kv_heads=2, vocab_size=64, dtype=DType.float32,
        hidden_size_multiple_of=8, rope_type=RoPEType.complex, rope_theta=1_000_000,
        layer_norm_eps=1e-6,
    )
    config.embedding_init_std = D**-0.5
    config.looped = LoopedConfig(**looped)
    torch.manual_seed(0)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"), max_seq_len=SEQ)
    assert isinstance(model, LoopedTransformer)
    return model


def _record_calls(model: LoopedTransformer) -> list:
    calls = []
    for key, block in model.blocks.items():
        block.register_forward_hook(
            lambda module, args, output, key=int(key): calls.append((key, torch.is_grad_enabled())))
    return calls


def _tokens(batch: int = 2) -> torch.Tensor:
    return torch.randint(0, 64, (batch, SEQ), generator=torch.Generator().manual_seed(1))


# --------------------------------------------------------------------------------------------------
# Execution order


@pytest.mark.parametrize("looped,n_layers,expected", [
    # Ouro: the whole stack 4 times.
    (dict(architecture="looped_transformer", loop_times=4), 3, [0, 1, 2] * 4),
    # A middle span.
    (dict(architecture="looped_transformer", loop_times=3, loop_start_layer=1, loop_end_layer=3), 4,
     [0] + [1, 2] * 3 + [3]),
    # Huginn P1 R2^5 C1.
    (dict(architecture="huginn", loop_times=5, prelude_layers=1, recurrent_layers=2, coda_layers=1), 4,
     [0] + [1, 2] * 5 + [3]),
])
def test_execution_order(looped, n_layers, expected):
    model = _model(n_layers, **looped)
    calls = _record_calls(model)
    with torch.no_grad():
        model.eval()(_tokens())
    assert [layer for layer, _ in calls] == expected
    assert list(model.execution_schedule) == expected


def test_hierarchical_state_orders():
    # shared_hl: H=2 cycles of L=3 low updates and one high update over the same span (1, 2).
    model = _model(4, architecture="huginn", loop_times=8, prelude_layers=1, recurrent_layers=2,
                   coda_layers=1, state_init="normal", hierarchical_state="shared_hl")
    calls = _record_calls(model)
    with torch.no_grad():
        model.eval()(_tokens())
    assert [layer for layer, _ in calls] == [0] + [1, 2] * 8 + [3]
    # split_hl with repeats 2: H = block 1, L = block 2; each cycle runs L (x2) three times, then H (x2).
    model = _model(4, architecture="huginn", prelude_layers=1, recurrent_layers=2, coda_layers=1,
                   state_init="normal", hierarchical_state="split_hl", split_module_repeats=2)
    calls = _record_calls(model)
    with torch.no_grad():
        model.eval()(_tokens())
    cycle = [2, 2] * 3 + [1, 1]
    assert [layer for layer, _ in calls] == [0] + cycle * 2 + [3]
    assert list(model.execution_schedule) == [0] + cycle * 2 + [3]


def test_truncated_backprop_runs_early_recurrences_without_gradients():
    model = _model(4, architecture="huginn", depth_control=True, loop_times=6, prelude_layers=1,
                   recurrent_layers=2, coda_layers=1, state_init="normal", backprop_depth=2)
    calls = _record_calls(model)
    model.train()(_tokens())
    core = [grad for layer, grad in calls if layer in (1, 2)]
    # Six recurrences of two blocks: the first four without gradients, the last two with them.
    assert core == [False] * 8 + [True] * 4
    assert calls[0] == (0, True) and calls[-1] == (3, True)


def test_truncated_gradient_equals_window_gradient():
    """The truncated gradient is the gradient through the last b recurrences with the earlier
    state held fixed."""
    model = _model(4, architecture="huginn", depth_control=True, loop_times=5, prelude_layers=1,
                   recurrent_layers=2, coda_layers=1, state_init="zero", backprop_depth=2)
    model.train()
    tokens = _tokens()
    model(tokens).float().square().mean().backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    # The same computation by hand: prelude, 3 recurrences detached, 2 with gradients, coda.
    h = model.embeddings(tokens)
    h = model.blocks["0"](h)
    e = h
    s = torch.zeros_like(e)
    for step in range(5):
        if step < 3:
            with torch.no_grad():
                s = model.blocks["2"](model.blocks["1"](model.input_injection(s, e)))
        else:
            s = model.blocks["2"](model.blocks["1"](model.input_injection(s, e)))
    model.lm_head(model.blocks["3"](s)).float().square().mean().backward()
    for name, grad in got.items():
        torch.testing.assert_close(grad, model.get_parameter(name).grad, rtol=1e-5, atol=1e-7, msg=name)


# --------------------------------------------------------------------------------------------------
# Injections


def test_diagonal_injection_is_the_release_formula():
    injection = DiagonalInjection(D, dt_init=0.7, a_init=1.3)
    injection.reset_parameters()
    assert torch.equal(injection.input_adapter.weight, torch.eye(D))
    torch.testing.assert_close(torch.nn.functional.softplus(injection.dt_bias), torch.full((D,), 0.7))
    h, e = torch.randn(3, D), torch.randn(3, D)
    torch.testing.assert_close(injection(h, e), math.exp(-0.7 * 1.3) * h + 0.7 * e)


def test_orthogonal_injection_keeps_the_input_component_fixed():
    injection = OrthogonalInjection(D, eps=0.0, dt_init=0.5 * math.log(5), a_init=1.0)
    injection.reset_parameters()
    h, e = torch.randn(4, D, dtype=torch.float64), torch.randn(4, D, dtype=torch.float64)
    injection = injection.double()
    out = injection(h, e)
    q = injection.decayed_state_and_update(h, e)[1]
    along = (out * q).sum(-1) / (q * q).sum(-1)
    torch.testing.assert_close(along, torch.ones(4, dtype=torch.float64))


def test_linear_injection_concatenates_state_then_input():
    injection = LinearInjection(D, bias=True, init_std=0.02)
    injection.reset_parameters()
    assert torch.count_nonzero(injection.adapter.bias) == 0
    h, e = torch.randn(2, D), torch.randn(2, D)
    w = injection.adapter.weight
    torch.testing.assert_close(injection(h, e), h @ w[:, :D].T + e @ w[:, D:].T)


# --------------------------------------------------------------------------------------------------
# Depth draws and the learned prior


def test_pln5_pmf_is_the_paper_runs_pmf():
    _, pmf = calibrated_capped_poisson_lognormal(5.0, 0.5, 64)
    assert max(abs(a - b) for a, b in zip(pmf, PAPER_PLN5_PMF)) < 1e-9
    assert abs(sum(i * p for i, p in enumerate(PAPER_PLN5_PMF, start=1)) - 5.0) < 1e-9


def test_depth_draws_are_seeded_by_the_forward_index():
    cdf = cumulative_distribution(PAPER_PLN5_PMF)
    draws = [sample_depth(cdf, step, antithetic=False) for step in range(2000)]
    assert draws == [sample_depth(cdf, step, antithetic=False) for step in range(2000)]
    assert 4.5 < sum(draws) / len(draws) < 5.5 and min(draws) == 1


def test_dry_run_does_not_advance_the_draws_and_the_prior_trains():
    model = _model(4, architecture="huginn", depth_control=True, loop_times=5, prelude_layers=1,
                   recurrent_layers=2, coda_layers=1, state_init="normal", sampling="poisson_lognormal",
                   backprop_depth=10, learned_depth_prior=True, depth_prior_entropy=0.01)
    model.train()
    tokens = _tokens()
    labels = torch.randint(0, 64, tokens.shape)
    model(tokens, labels=labels).loss.backward()
    model.post_batch(dry_run=True)
    assert model.iteration_step == 0 and not model.depth_prior.pending_depths
    first_draw = model.depth_prior.sample()
    model.depth_prior.pending_depths.clear()
    model.post_batch(dry_run=True)
    for _ in range(40):  # past the prior's 32 burn-in steps
        for _ in range(2):  # two accumulation steps per optimizer step
            model(tokens, labels=labels).loss.backward()
        model.post_batch()
        model.post_optim_step()
    assert model.depth_prior.updates == 40
    assert model.depth_prior.state_dict()["logits"].shape == (64,)
    assert first_draw >= 1
    assert model.training_state_dict()["iteration_step"] == 80


# --------------------------------------------------------------------------------------------------
# Parameters, initialization, configs


def test_initialization_follows_the_release():
    model = _model(4, architecture="huginn", loop_times=5, prelude_layers=1, recurrent_layers=2,
                   coda_layers=1, state_init="normal")
    head = model.lm_head.w_out.weight
    assert abs(head.std().item() - D**-0.5) < 0.2 * D**-0.5
    assert head.abs().max().item() <= 3 * D**-0.5 + 1e-6
    emb = model.embeddings.weight
    assert abs(emb.std().item() - D**-0.5) < 0.2 * D**-0.5
    assert torch.equal(model.input_injection.input_adapter.weight, torch.eye(D))


def test_evaluation_state_is_deterministic():
    model = _model(4, architecture="huginn", loop_times=5, prelude_layers=1, recurrent_layers=2,
                   coda_layers=1, state_init="normal").eval()
    with torch.no_grad():
        torch.testing.assert_close(model(_tokens()), model(_tokens()), rtol=0, atol=0)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("part*/*.json")), ids=lambda p: p.stem)
def test_recipe_configs_build_with_matching_parameter_counts(path):
    import sys

    sys.path.insert(0, str(ROOT))
    from config_utils.model_config import load_llama_like_kwargs

    kwargs, raw = load_llama_like_kwargs(path, tokenizer_vocab_size=50280)
    kwargs["dtype"] = DType.float32
    config = TransformerConfig.llama_like(**kwargs)
    config.looped = LoopedConfig.from_dict(raw["looped"])
    config.looped.validate(config.n_layers)
    model = config.build(init_device="meta")
    assert model.num_params == config.num_params
    block = sum(p.numel() for p in model.blocks["0"].parameters())
    width = config.d_model
    # Table 6 of Part II and Section 3.1 of Part I: about 25.95M parameters per block at width 1536.
    if width == 1536:
        assert abs(block - 25.95e6) < 0.01e6
    injection = config.looped.num_params(width)
    total_blocks = sum(p.numel() for p in model.blocks.parameters())
    assert model.num_params - total_blocks - 2 * 50280 * width - width == injection


def test_recipe_index_lists_every_config():
    index = json.loads((CONFIGS / "recipes.json").read_text())
    assert len(index) == 30 and all((ROOT / "looped" / v["config"]).exists() for v in index.values())
