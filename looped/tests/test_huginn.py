"""Tests of the Huginn-0125 configuration (``looped/configs/huginn/huginn_0125.json``): the release's size and
blocks, and a small model of the same configuration on CPU.

Run from the repository root:
    PYTHONPATH=pretrain/OLMo-core/src:. python -m pytest -q looped/tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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
    return torch.randint(0, VOCAB, (batch, SEQ), generator=generator)


def test_release_config_builds_tied_sandwich_model():
    config = build_olmo(CONFIG, dtype=DType.float32)
    model = config.build(init_device="meta")
    assert all(isinstance(block, SandwichTransformerBlock) for block in model.blocks.values())
    assert model.lm_head.w_out.weight is model.embeddings.weight
    assert model.num_params == config.num_params
    assert 3.5e9 < model.num_params < 3.6e9


@torch.no_grad()
def test_small_huginn_runs_the_recurrence_and_is_causal(tmp_path):
    model = _model(tmp_path)
    ids = _inputs()
    model.initialize_recurrent_state = lambda embeds, cache_len=0: torch.ones_like(embeds) * 0.01
    out = model(ids)
    assert out.shape == (2, SEQ, VOCAB) and torch.isfinite(out).all()
    changed = ids.clone()
    changed[:, 6] = (changed[:, 6] + 1) % VOCAB
    later = model(changed)
    assert torch.allclose(later[:, :6], out[:, :6], atol=1e-5)
    model.looped.loop_times = 2
    assert not torch.allclose(model(ids), out, atol=1e-3)
