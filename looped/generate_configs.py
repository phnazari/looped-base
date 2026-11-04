#!/usr/bin/env python3
"""Generate the model configs of the looped models of *Towards Looped Models Done Right*.

Every recipe here is a model of the official release, xLLM-Loop (https://github.com/ifm-ai/xllm-loop):
the dense recipes of ``xllm/paper_part1/recipes.py`` (Part I) and every recipe of
``xllm/paper_part2/recipes.py`` (Part II), with the same names. Only the model is ported; data,
tokenizer, sequence length, optimizer and schedule are this repository's training recipe
(``pretrain/train.py``). The release's peak learning rate of each recipe is listed in
``configs/recipes.json`` for reference.

Not ported: Part I's MoE recipes (moe_*), whose router (sigmoid top-2 with renormalized scores and
the release's "dot" balance loss) differs from OLMo-core's.

Run ``python looped/generate_configs.py`` after changing this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "configs"

# The shared numerics of both releases: Llama blocks with GQA, SwiGLU, complex (interleaved) RoPE
# with base 1e6, RMSNorm eps 1e-6, untied embeddings; init std 0.02 for the block matrices and
# d**-0.5 for the embeddings and the output head (the output head is set by the model).
VOCAB_SIZE = 50280  # this repository's tokenizer; the releases use their own


def _llama(d_model: int, heads: int, kv_heads: int, ffn: int, layers: int) -> dict:
    return {
        "architectures": ["LLaMAForCausalLM"],
        "attention_bias": False,
        "bos_token_id": None,
        "eos_token_id": 50279,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": d_model,
        "intermediate_size": ffn,
        "initializer_range": 0.02,
        "embedding_init_std": d_model**-0.5,
        "max_sequence_length": 2048,
        "model_type": "llama",
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "pad_token_id": 1,
        "rope_theta": 1_000_000,
        "rope_type": "complex",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": VOCAB_SIZE,
    }


# ---------------------------------------------------------------------------------------------------
# Part I (dense): width 1536, 24 query / 6 KV heads, FFN 4352; 112 logical layers.

PART1_LR = 6e-4  # one fixed learning rate for every recipe


def _part1(layers: int, looped: dict) -> dict:
    return {**_llama(1536, 24, 6, 4352, layers), "looped": looped}


_HUGINN_P1 = {"architecture": "huginn", "loop_times": 8, "prelude_layers": 8, "recurrent_layers": 12,
              "coda_layers": 8, "state_init": "normal", "input_injection": "diagonal",
              "diagonal_dt_init": 1.0, "diagonal_a_init": 1.0}

PART1 = {
    "dense_d28": _part1(28, {"architecture": "looped_transformer", "loop_times": 1}),
    "dense_d112": _part1(112, {"architecture": "looped_transformer", "loop_times": 1}),
    "dense_ouro": _part1(28, {"architecture": "looped_transformer", "loop_times": 4}),
    "dense_middle": _part1(28, {**_HUGINN_P1, "state_init": "input", "input_injection": "none"}),
    "dense_ouro_raw_injection": _part1(28, {"architecture": "looped_transformer", "loop_times": 4,
                                            "loop_input_injection": "diagonal"}),
    "dense_middle_prelude_injection": _part1(28, {**_HUGINN_P1, "state_init": "input"}),
    "dense_huginn": _part1(28, _HUGINN_P1),
    "dense_shared_hl": _part1(28, {**_HUGINN_P1, "hierarchical_state": "shared_hl"}),
    "dense_split_hl": _part1(28, {**_HUGINN_P1, "hierarchical_state": "split_hl"}),
    "dense_split_hl_x2": _part1(28, {**_HUGINN_P1, "hierarchical_state": "split_hl",
                                     "split_module_repeats": 2}),
}

# ---------------------------------------------------------------------------------------------------
# Part II: prelude/core/coda of 1/2/1 blocks at three widths, trained at R=5.

SCALES = {  # d_model, heads, KV heads, FFN
    "s": (1536, 24, 6, 4352),
    "m": (3072, 48, 12, 8704),
    "l": (6144, 96, 24, 17408),
}

_HUGINN_P2 = {"architecture": "huginn", "depth_control": True, "loop_times": 5, "prelude_layers": 1,
              "recurrent_layers": 2, "coda_layers": 1, "state_init": "normal",
              "input_injection": "diagonal", "diagonal_dt_init": 1.0, "diagonal_a_init": 1.0,
              "sampling": "fixed", "backprop_depth": 5}
# Fixed PLN-5: capped Poisson-lognormal depth with mean 5 (sigma 0.5, cap 64), backpropagating the last 10.
PLN5 = {"sampling": "poisson_lognormal", "poisson_lognormal_mean": 5.0, "poisson_lognormal_sigma": 0.5,
        "poisson_lognormal_max": 64, "backprop_depth": 10}
PARCAE_DT_INIT = 0.5 * math.log(5)
INJECTIONS = {
    # W[h; e] + b with a parameter-free RMSNorm after every recurrence.
    "huginn_linear": {"input_injection": "linear", "recurrent_exit_norm": True},
    # Diagonal injection of the RMS-normalized prelude output, dt initialized at 1/2 log 5.
    "parcae": {"diagonal_dt_init": PARCAE_DT_INIT, "prelude_norm": True},
    # Diagonal injection whose decayed state drops its component along the injected update.
    "orthoinj": {"diagonal_dt_init": PARCAE_DT_INIT, "orthogonal_injection": True},
}


def _part2(scale: str, **looped) -> dict:
    return {**_llama(*SCALES[scale], 4), "looped": {**_HUGINN_P2, **looped}}


def _learned(entropy: float) -> dict:
    return {**PLN5, "learned_depth_prior": True, "depth_prior_entropy": entropy}


# name: (release table row, release learning rate, config), in the release's table order.
PART2 = {
    "s_fixed_r5": ("Table 2, S, Fixed R=5", 1.5e-3, _part2("s")),
    "m_fixed_r5": ("Table 2, M, Fixed R=5", 1.5e-3, _part2("m")),
    "l_fixed_r5": ("Table 2, L, Fixed R=5", 1e-3, _part2("l")),
    "s_fixed_pln5": ("Table 2, S, Fixed PLN-5", 1.5e-3, _part2("s", **PLN5)),
    "m_fixed_pln5": ("Table 2, M, Fixed PLN-5", 1.5e-3, _part2("m", **PLN5)),
    "l_fixed_pln5": ("Table 2, L, Fixed PLN-5", 5e-4, _part2("l", **PLN5)),
    "s_learned_entropy0": ("Table 2, S, Learned lambda_H=0", 2e-3, _part2("s", **_learned(0.0))),
    "s_learned_entropy0p01": ("Table 2, S, Learned lambda_H=0.01", 2e-3, _part2("s", **_learned(0.01))),
    "m_learned_entropy0": ("Table 2, M, Learned lambda_H=0", 1.5e-3, _part2("m", **_learned(0.0))),
    "m_learned_entropy0p01": ("Table 2, M, Learned lambda_H=0.01", 1.5e-3, _part2("m", **_learned(0.01))),
    "l_learned_entropy0": ("Table 2, L, Learned lambda_H=0", 1e-3, _part2("l", **_learned(0.0))),
    "l_learned_entropy0p01": ("Table 2, L, Learned lambda_H=0.01", 1e-3, _part2("l", **_learned(0.01))),
    "s_huginn_linear": ("Table 3, S, Huginn-Linear", 1e-3, _part2("s", **PLN5, **INJECTIONS["huginn_linear"])),
    "s_parcae": ("Table 3, S, Parcae-Decay", 1.5e-3, _part2("s", **PLN5, **INJECTIONS["parcae"])),
    "s_orthoinj": ("Table 3, S, OrthoInj", 2e-3, _part2("s", **PLN5, **INJECTIONS["orthoinj"])),
    "m_huginn_linear": ("Table 3, M, Huginn-Linear", 7.5e-4, _part2("m", **PLN5, **INJECTIONS["huginn_linear"])),
    "m_parcae": ("Table 3, M, Parcae-Decay", 1e-3, _part2("m", **PLN5, **INJECTIONS["parcae"])),
    "m_orthoinj": ("Table 3, M, OrthoInj", 1.5e-3, _part2("m", **PLN5, **INJECTIONS["orthoinj"])),
    # The dense baseline D4: the four blocks run once in sequence.
    "s_d4": ("Table 4, S, D4", 2e-3, {**_llama(*SCALES["s"], 4),
                                      "looped": {"architecture": "looped_transformer", "loop_times": 1}}),
    "m_d4": ("Table 4, M, D4", 1e-3, {**_llama(*SCALES["m"], 4),
                                      "looped": {"architecture": "looped_transformer", "loop_times": 1}}),
}


# ---------------------------------------------------------------------------------------------------
# Not in the release. More loops: Fixed PLN-5 at a larger mean depth (10 or 20), backpropagating the
# last 2 x mean recurrences as PLN-5 does (10).


def _deeper(config: dict, mean: int) -> dict:
    return {**config, "looped": {**config["looped"], "poisson_lognormal_mean": float(mean), "backprop_depth": 2 * mean}}


DEPTH = {f"s_fixed_pln{mean}": _deeper(PART2["s_fixed_pln5"][2], mean) for mean in (10, 20)}


def main() -> None:
    index = {}
    for group, configs in (("depth", DEPTH),):
        directory = OUTPUT / group
        directory.mkdir(parents=True, exist_ok=True)
        for name, config in configs.items():
            (directory / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"wrote {len(configs)} configs to {directory}")
    for part, recipes in (("part1", {name: (None, PART1_LR, cfg) for name, cfg in PART1.items()}),
                          ("part2", PART2)):
        directory = OUTPUT / part
        directory.mkdir(parents=True, exist_ok=True)
        for name, (row, lr, config) in recipes.items():
            path = directory / f"{name}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            index[name] = {"config": str(path.relative_to(ROOT)), "release_lr": lr,
                           **({"release_row": row} if row else {})}
    (OUTPUT / "recipes.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(index)} configs to {OUTPUT}")


if __name__ == "__main__":
    main()
