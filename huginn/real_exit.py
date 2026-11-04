"""Real exits for the OLMo-core Huginn: token-by-token decoding in which a token runs the core only until it
exits (a fixed count, the exit head, or Huginn's latent-diff rule), and later tokens read the keys and values of
its last recurrence (Huginn's "latest" cache lookup). The parallel forward with ``LoopedConfig.frozen_kv`` computes
the same. Run from the repository root, one GPU:

    python huginn/real_exit.py --checkpoint <model.pt or run/stepN> --check
    python huginn/real_exit.py --checkpoint <model.pt> --generate --prompt "..." --latent-threshold 0.1
    python huginn/real_exit.py --checkpoint <run/stepN> --generate --exit-head

``--check`` compares the decoder's logits with the parallel forward (FP32, random per-token counts), with and
without ``frozen_kv``. ``--generate`` continues the prompt greedily, every token exiting at ``--count``, by the exit
head (``--exit-head``) or by the latent-diff rule (``--latent-threshold``), and reports the recurrences run and the
time per token.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from convert import CONFIG, build_olmo  # noqa: E402  (sets up the paths)
from hf_load import REPO  # noqa: E402
from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.checkpoint import load_model_and_optim_state  # noqa: E402

TEXT = ("The capital of France is Paris. The capital of Germany is Berlin, and the capital of Italy is Rome. "
        "Rivers carry water from the mountains to the sea, and along their banks towns grew into cities.")


def load(checkpoint: Path, dtype: torch.dtype, exit_head: bool):
    """A converted ``model.pt`` (``convert.py``) or a training checkpoint directory (``run/stepN``)."""
    config = build_olmo(CONFIG, dtype=DType.float32)
    config.looped.exit_head = exit_head
    model = config.build(init_device="cuda")
    if checkpoint.suffix == ".pt":
        missing, unexpected = model.load_state_dict(
            torch.load(checkpoint, map_location="cuda", mmap=True, weights_only=True), strict=False)
        if unexpected or any(not key.startswith("exit_head.") for key in missing):
            raise RuntimeError(f"missing {missing}, unexpected {unexpected}")
    else:
        load_model_and_optim_state(checkpoint / "model_and_optim", model, strict=False)
    return model.to(dtype).eval()


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REPO)


class RealExitDecoder:
    """One token at a time with key/value caches per block, and per recurrence for the core blocks. A token
    runs recurrences until it exits; its last keys and values then fill the later recurrences' slots."""

    def __init__(self, model, max_len: int, depth: int):
        self.model, self.depth = model, depth
        looped = model.looped
        blocks = [model.blocks[str(i)] for i in range(model.n_layers)]
        prelude, core = looped.prelude_layers, looped.recurrent_count(model.n_layers)
        self.prelude, self.core, self.coda = blocks[:prelude], blocks[prelude:prelude + core], blocks[prelude + core:]
        attention = blocks[0].attention
        shape = (max_len, attention.n_kv_heads, attention.head_dim)
        kw = dict(device="cuda", dtype=next(model.parameters()).dtype)
        self.prelude_kv = torch.zeros(len(self.prelude), 2, *shape, **kw)
        self.core_kv = torch.zeros(len(self.core), depth, 2, *shape, **kw)
        self.coda_kv = torch.zeros(len(self.coda), 2, *shape, **kw)
        self.pos = 0

    @staticmethod
    def _attend(attention, x, pos: int, kv: torch.Tensor):
        """``kv``: this block's cache (2 x max_len x heads x head_dim); position ``pos`` is written."""
        head_dim = attention.head_dim
        q = attention.w_q(x).view(1, 1, -1, head_dim)
        k = attention.w_k(x).view(1, 1, -1, head_dim)
        v = attention.w_v(x).view(1, 1, -1, head_dim)
        q, k = attention.rope(q, k, head_first=False, start_pos=pos)
        kv[0, pos], kv[1, pos] = k[0, 0], v[0, 0]
        keys, values = kv[0, : pos + 1].transpose(0, 1)[None], kv[1, : pos + 1].transpose(0, 1)[None]
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys, values)
        return attention.w_out(out.transpose(1, 2).reshape(1, 1, -1))

    def _block(self, block, x, pos: int, kv: torch.Tensor):
        """The sandwich block (``SandwichTransformerBlock.forward``) on one token."""
        h = block.post_attention_norm(
            block.attention_residual_stream(x, self._attend(block.attention, block.attention_norm(x), pos, kv)))
        return block.post_feed_forward_norm(
            block.feed_forward_residual_stream(h, block.feed_forward(block.feed_forward_norm(h))))

    @torch.no_grad()
    def step(self, token: int, state: torch.Tensor, count: Optional[int] = None, may_exit: bool = False,
             threshold: float = 0.5, latent_threshold: Optional[float] = None):
        """Feed one token with its initial state (1 x 1 x d). The core runs ``count`` recurrences, or until
        the exit head fires if ``may_exit``, or until the state changes by less than ``latent_threshold``
        relative to the last recurrence (Huginn's ``latent-diff`` exit), else the full depth. Returns the logits
        and the count run."""
        model, pos = self.model, self.pos
        self.last_changes = []  # latent-diff values of this token's recurrences
        x = model.embeddings(torch.tensor([[token]], device="cuda"))
        if model.embed_scale is not None:
            x = x * model.embed_scale
        for i, block in enumerate(self.prelude):
            x = self._block(block, x, pos, self.prelude_kv[i])
        inputs = model._recurrence_input(x)
        ran = self.depth
        for step in range(self.depth):
            previous = state
            if model.input_injection is not None:
                state = model.input_injection(state, inputs)
            for j, block in enumerate(self.core):
                state = self._block(block, state, pos, self.core_kv[j, step])
            done = step + 1
            if count is not None and done >= count:
                ran = done
                break
            if latent_threshold is not None:
                change = ((state - previous).float().norm() / state.float().norm()).item()
                self.last_changes.append(change)
                if change < latent_threshold:
                    ran = done
                    break
            if count is None and may_exit:
                if model.exit_head(model.lm_head.norm(state)).float().sigmoid().item() > threshold:
                    ran = done
                    break
        if ran < self.depth:
            self.core_kv[:, ran:, :, pos] = self.core_kv[:, ran - 1:ran, :, pos]
        h = model.lm_head.norm(state) if model.looped.recurrence_final_norm else state
        for i, block in enumerate(self.coda):
            h = self._block(block, h, pos, self.coda_kv[i])
        self.pos += 1
        return model.lm_head(h)[0, -1], ran


def initial_states(length: int, d: int, std: float, scale: float, seed: int, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.empty(1, length, d, device="cuda")
    torch.nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std, generator=generator)
    return (state * scale).to(dtype)


@torch.no_grad()
def check(model, tok, depth: int = 8):
    """Decoder vs parallel forward on ``TEXT`` with random counts in 1..depth (FP32)."""
    ids = tok(TEXT, return_tensors="pt", add_special_tokens=True).input_ids.cuda()
    length = ids.shape[1]
    counts = torch.randint(1, depth + 1, ids.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    looped = model.looped
    states = initial_states(length, model.d_model, looped.state_init_std, model.embed_scale or 1.0, 1,
                            next(model.parameters()).dtype)
    model.initialize_recurrent_state = lambda embeds, cache_len=0, seeds=None: states
    looped.loop_times = depth
    decoder = RealExitDecoder(model, length, depth)
    decoded = torch.stack([decoder.step(int(ids[0, t]), states[:, t:t + 1], count=int(counts[0, t]))[0]
                           for t in range(length)])
    out = {}
    for frozen in (True, False):
        looped.frozen_kv = frozen
        parallel = model(ids, loop_counts=counts)[0].float()
        rel = ((decoded.float() - parallel).abs().max() / parallel.abs().max()).item()
        out[f"frozen_kv={frozen}"] = rel
        print(f"parallel forward frozen_kv={frozen}: max relative logit difference to the decoder {rel:.3e}, "
              f"argmax agreement {(decoded.argmax(-1) == parallel.argmax(-1)).float().mean().item():.3f}", flush=True)
    return out


@torch.no_grad()
def generate(model, tok, prompt: str, new_tokens: int, count: Optional[int] = None, exit_head: bool = False,
             latent_threshold: Optional[float] = None):
    """Continue ``prompt`` greedily. The prompt runs the full depth; every new token exits at ``count``, by the
    exit head or by the latent-diff rule (the full depth if none is given)."""
    depth = model.looped.loop_times
    ids = tok(prompt, add_special_tokens=True).input_ids
    total = len(ids) + new_tokens
    decoder = RealExitDecoder(model, total, depth)
    states = initial_states(total, model.d_model, model.looped.state_init_std, model.embed_scale or 1.0, 2,
                            next(model.parameters()).dtype)
    for t, token in enumerate(ids):
        logits, _ = decoder.step(token, states[:, t:t + 1])
    generated, counts = [], []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for t in range(len(ids), total):
        token = int(logits.argmax())
        generated.append(token)
        logits, ran = decoder.step(token, states[:, t:t + 1], count=count, may_exit=exit_head,
                                   latent_threshold=latent_threshold)
        counts.append(ran)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / new_tokens
    result = {"count": count, "exit head": exit_head, "latent threshold": latent_threshold,
              "mean recurrences": sum(counts) / len(counts), "ms per token": 1000 * seconds,
              "text": tok.decode(generated)}
    print(json.dumps(result), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="model.pt or run/stepN")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--depth", type=int, default=32, help="The most recurrences a token runs.")
    exits = ap.add_mutually_exclusive_group()
    exits.add_argument("--count", type=int, help="Every new token exits after this many recurrences.")
    exits.add_argument("--exit-head", action="store_true", help="Exit by the exit head (a checkpoint with one).")
    exits.add_argument("--latent-threshold", type=float, help="Exit once the state changes by less than this.")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tok = load_tokenizer()
    checkpoint = args.checkpoint.expanduser()
    results = {"checkpoint": str(checkpoint)}
    if args.check:
        model = load(checkpoint, torch.float32, exit_head=False)
        results["check"] = check(model, tok)
        del model
        torch.cuda.empty_cache()
    if args.generate:
        model = load(checkpoint, torch.bfloat16, exit_head=args.exit_head)
        model.looped.loop_times = args.depth
        results["generate"] = generate(model, tok, args.prompt, args.new_tokens, count=args.count,
                                       exit_head=args.exit_head, latent_threshold=args.latent_threshold)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
