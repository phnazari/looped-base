"""Tests of what this repository adds to the Huginn architecture: per-token loop counts, real exits
(``frozen_kv``), the exit head and Huginn's latent-diff exit rule, and seeded initial states. A small model
with the configuration of Huginn-0125 (``looped/configs/huginn/huginn_0125.json``), on CPU.

Run from the repository root:
    PYTHONPATH=pretrain/OLMo-core/src:. python -m pytest -q looped/tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.transformer import SandwichTransformerBlock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "huginn"))
from convert import CONFIG, build_olmo  # noqa: E402

DEPTH, SEQ, VOCAB = 6, 12, 128


def _small_config(tmp_path: Path, **looped):
    raw = json.loads(CONFIG.read_text())
    raw.update(hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
               vocab_size=VOCAB, embed_scale=8.0)
    raw["looped"].update(loop_times=DEPTH, backprop_depth=None, **looped)
    path = tmp_path / "huginn_small.json"
    path.write_text(json.dumps(raw))
    return build_olmo(path, dtype=DType.float32)


def _model(tmp_path: Path, **looped):
    config = _small_config(tmp_path, **looped)
    torch.manual_seed(0)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"), max_seq_len=SEQ)
    return model.eval()


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (batch, SEQ), generator=generator)
    seeds = torch.arange(batch) + 7
    return ids, seeds


def test_release_config_builds_tied_sandwich_model():
    config = build_olmo(CONFIG, dtype=DType.float32)
    model = config.build(init_device="meta")
    assert all(isinstance(block, SandwichTransformerBlock) for block in model.blocks.values())
    assert model.lm_head.w_out.weight is model.embeddings.weight
    assert model.num_params == config.num_params
    assert 3.5e9 < model.num_params < 3.6e9


@torch.no_grad()
def test_full_loop_counts_and_frozen_kv_match_the_plain_forward(tmp_path):
    model = _model(tmp_path)
    ids, seeds = _inputs()
    plain = model(ids, state_seeds=seeds)
    full = torch.full_like(ids, DEPTH)
    assert torch.allclose(model(ids, loop_counts=full, state_seeds=seeds), plain, atol=1e-5)
    model.looped.frozen_kv = True
    assert torch.allclose(model(ids, loop_counts=full, state_seeds=seeds), plain, atol=1e-5)


@torch.no_grad()
def test_loop_counts_change_only_the_shortened_tokens_and_later_ones(tmp_path):
    model = _model(tmp_path)
    ids, seeds = _inputs()
    counts = torch.full_like(ids, DEPTH)
    plain = model(ids, loop_counts=counts, state_seeds=seeds)
    counts[:, 5] = 2
    short = model(ids, loop_counts=counts, state_seeds=seeds)
    assert torch.allclose(short[:, :5], plain[:, :5], atol=1e-5)  # causal: earlier tokens are unchanged
    assert not torch.allclose(short[:, 5], plain[:, 5], atol=1e-3)
    with pytest.raises(ValueError, match="loop_counts"):
        model(ids, loop_counts=counts + DEPTH, state_seeds=seeds)


def test_seeded_states_do_not_depend_on_length_or_batch(tmp_path):
    model = _model(tmp_path)
    embeds = torch.zeros(2, SEQ, model.d_model)
    seeds = torch.tensor([3, 4])
    both = model.initialize_recurrent_state(embeds, seeds=seeds)
    first_half = model.initialize_recurrent_state(embeds[1:, : SEQ // 2], seeds=seeds[1:])
    assert torch.equal(both[1:, : SEQ // 2], first_half)
    assert not torch.equal(both[0], both[1])
    std = model.looped.state_init_std * model.embed_scale
    assert both.abs().max() <= 3 * std + 1e-6


@torch.no_grad()
def test_latent_rule_halts_masked_tokens(tmp_path):
    model = _model(tmp_path, exit_rule="latent", exit_latent_threshold=1.0)
    ids, seeds = _inputs()
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 4:] = True
    model(ids, loop_counts=torch.full_like(ids, DEPTH), exit_mask=mask, exit_threshold=0.5, state_seeds=seeds)
    counts = model.last_exit_counts
    assert (counts[~mask] == DEPTH).all()
    assert ((counts[mask] >= 1) & (counts[mask] <= DEPTH)).all()
    assert (counts[mask] < DEPTH).any()  # a threshold of a 100% change fires early


def test_exit_head_trains_on_exit_targets(tmp_path):
    model = _model(tmp_path, exit_head=True).train()
    ids, seeds = _inputs()
    targets = torch.zeros_like(ids)
    targets[:, 3:] = 4
    output = model(ids, labels=ids, loop_counts=torch.full_like(ids, DEPTH), exit_targets=targets,
                   state_seeds=seeds)
    output.loss.backward()
    assert model.exit_head.weight.grad is not None and model.exit_head.weight.grad.abs().sum() > 0
    metrics = model.compute_auxiliary_metrics()
    assert 0 < metrics["exit/bce"][0] and 0 <= metrics["exit/exact"][0] <= 1
    # The head starts silent: it fires at no recurrence, so no token exits at its target.
    assert metrics["exit/exact"][0] == 0
