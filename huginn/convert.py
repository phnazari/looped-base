"""Convert tomg-group-umd/huginn-0125 into this repository's OLMo-core looped model, and check parity.

The OLMo-core model is built from `looped/configs/huginn/huginn_0125.json` (sandwich blocks, query/key biases,
linear injection, truncated-normal initial state, the final norm also after the recurrence). Run from the
repository root:

    python huginn/convert.py --out <dir>   # writes <dir>/model.pt (FP32)

Parity: both models in FP32 on one GPU, the same initial state, logits compared at several loop counts.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "pretrain" / "OLMo-core" / "src", Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from config_utils.model_config import load_llama_like_kwargs  # noqa: E402
from hf_load import REPO, load_huginn  # noqa: E402
from olmo_core.config import DType  # noqa: E402
from olmo_core.nn.looped import LoopedConfig  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

CONFIG = REPO_ROOT / "looped/configs/huginn/huginn_0125.json"
D, PRELUDE, CORE = 5280, 2, 4


def build_olmo(config_path: Path = CONFIG, dtype: DType = DType.float32):
    import json

    raw = json.loads(config_path.read_text())
    kwargs, _ = load_llama_like_kwargs(config_path, tokenizer_vocab_size=raw["vocab_size"])
    kwargs["dtype"] = dtype
    if kwargs.get("feed_forward") is not None:
        kwargs["feed_forward"].dtype = dtype
    config = TransformerConfig.llama_like(**kwargs)
    config.looped = LoopedConfig.from_dict(raw["looped"])
    config.looped.validate(config.n_layers)
    return config


def convert_state_dict(hf: dict) -> dict:
    out = {"embeddings.weight": hf["transformer.wte.weight"], "lm_head.norm.weight": hf["transformer.ln_f.weight"],
           "input_injection.adapter.weight": hf["transformer.adapter.weight"]}
    out["lm_head.w_out.weight"] = out["embeddings.weight"]  # tied
    groups = [("prelude", 0, PRELUDE), ("core_block", PRELUDE, CORE), ("coda", PRELUDE + CORE, 2)]
    for group, offset, count in groups:
        for i in range(count):
            src, dst = f"transformer.{group}.{i}.", f"blocks.{offset + i}."
            q, k, v = hf[src + "attn.Wqkv.weight"].split(D, dim=0)
            qb, kb = hf[src + "attn.qk_bias"].reshape(2, -1)
            w1, w3 = hf[src + "mlp.fc.weight"].chunk(2, dim=0)
            out.update({
                dst + "attention.w_q.weight": q, dst + "attention.w_q.bias": qb,
                dst + "attention.w_k.weight": k, dst + "attention.w_k.bias": kb,
                dst + "attention.w_v.weight": v,
                dst + "attention.w_out.weight": hf[src + "attn.proj.weight"],
                dst + "feed_forward.w1.weight": w1, dst + "feed_forward.w3.weight": w3,
                dst + "feed_forward.w2.weight": hf[src + "mlp.proj.weight"],
                dst + "attention_norm.weight": hf[src + "norm_1.weight"],
                dst + "post_attention_norm.weight": hf[src + "norm_2.weight"],
                dst + "feed_forward_norm.weight": hf[src + "norm_3.weight"],
                dst + "post_feed_forward_norm.weight": hf[src + "norm_4.weight"],
            })
    return {key: value.contiguous() for key, value in out.items()}


def load_hf_state_dict() -> dict:
    snapshot = Path(snapshot_download(REPO, allow_patterns=["*.safetensors", "*.json"]))
    state = {}
    for shard in sorted(snapshot.glob("model-*.safetensors")):
        state.update(load_file(shard))
    return state


@torch.no_grad()
def parity(olmo, hf_model, tok, steps=(1, 4, 32)):
    text = "The capital of France is Paris. The capital of Germany is Berlin, and the capital of Italy is"
    ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    e_shape = (1, ids.shape[1], D)
    torch.manual_seed(0)
    s0 = torch.randn(e_shape, device="cuda") * 0.0087 * 72.66
    hf_model.initialize_state = lambda embeds, scale=1.0: s0.clone()
    olmo.initialize_recurrent_state = lambda embeds, cache_len=0: s0.to(embeds.dtype)
    worst = 0.0
    for r in steps:
        ref = hf_model(ids, num_steps=torch.tensor([r, 0])).logits.float()
        olmo.looped.loop_times = r
        got = olmo(ids).float()
        diff = (got - ref).abs().max().item()
        rel = diff / ref.abs().max().item()
        same = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"r={r:3d}  max|dlogit|={diff:.3e}  rel={rel:.3e}  argmax agree={same:.4f}  "
              f"next={tok.decode(ref[0, -1].argmax())!r}/{tok.decode(got[0, -1].argmax())!r}", flush=True)
        worst = max(worst, rel)

    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-parity", action="store_true")
    args = ap.parse_args()

    state = convert_state_dict(load_hf_state_dict())
    olmo = build_olmo().build(init_device="cpu")
    missing, unexpected = olmo.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"missing {missing}, unexpected {unexpected}")
    assert olmo.lm_head.w_out.weight is olmo.embeddings.weight, "embeddings are not tied"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(olmo.state_dict(), args.out / "model.pt")
    print(f"saved {args.out / 'model.pt'}", flush=True)

    if not args.no_parity:
        olmo = olmo.cuda().eval()
        hf_model, tok = load_huginn(dtype=torch.float32)
        worst = parity(olmo, hf_model, tok)
        print(f"worst relative logit difference {worst:.3e}")


if __name__ == "__main__":
    main()
