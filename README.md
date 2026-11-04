# looped-base

Looped language models on OLMo-core: training and Huginn-0125.

- `pretrain/OLMo-core/`: OLMo-core v2.5.0 with the looped models (`src/olmo_core/nn/looped/`) and Huginn's
  sandwich block.
  - The looped models of *Towards Looped Models Done Right* (Parts I and II) are ported from xLLM-Loop:
    the Ouro-style loop, Huginn with H/L states, and depth control (fixed, Poisson-lognormal or learned depth,
    truncated backprop, and the linear, Parcae and orthogonal injections).
  - Huginn-0125: the sandwich block, query/key biases, a truncated-normal initial state and the final norm
    after the recurrence (`looped/configs/huginn/huginn_0125.json`).
- `pretrain/train.py`, `pretrain/launch.sh`, `pretrain/configs/`: the pretraining recipe (FineWeb-Edu, 512 × 2048
  tokens per step, AdamW with cosine decay).
- `looped/configs/`: model configs. `part1/` and `part2/` hold the release recipes (their learning rates are in
  `recipes.json`), `depth/` holds deeper PLN variants, and `huginn/` holds Huginn-0125.
  `looped/generate_configs.py` writes them.
- `huginn/convert.py`: converts the HF checkpoint `tomg-group-umd/huginn-0125` into this model and checks
  logit parity.
- `eval/`: zero-shot evaluation with lm-eval.
- `data_utils/`: tokenization.

## Install

```bash
uv sync --frozen
source .venv/bin/activate
```

## Train

```bash
export DATA_DIR=... WANDB_ENTITY=... CHECKPOINT_DIR=...
bash pretrain/launch.sh looped/configs/part2/s_fixed_pln5.json ablation 0.0015
```

## Huginn-0125

```bash
python huginn/convert.py --out checkpoints/huginn-0125               # model.pt + parity check
```

## Test

```bash
PYTHONPATH=pretrain/OLMo-core/src:. python -m pytest -q looped/tests
```
