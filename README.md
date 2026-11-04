# looped-base

Looped language models on OLMo-core: training, Huginn-0125 and real per-token exits.

- `pretrain/OLMo-core/`: OLMo-core v2.5.0 with the looped models (`src/olmo_core/nn/looped/`) and Huginn's
  sandwich block.
  - The looped models of *Towards Looped Models Done Right* (Parts I and II) are ported from xLLM-Loop:
    the Ouro-style loop, Huginn with H/L states, and depth control (fixed, Poisson-lognormal or learned depth,
    truncated backprop, and the linear, Parcae and orthogonal injections).
  - Huginn-0125 support:
    - per-token loop counts (`loop_counts`)
    - real exits that reuse an exited token's last keys and values (`frozen_kv`)
    - an exit head
    - Huginn's latent-diff exit rule (`exit_rule="latent"`)
    - seeded initial states (`state_seeds`)
- `pretrain/train.py`, `pretrain/launch.sh`, `pretrain/configs/`: the pretraining recipe (FineWeb-Edu, 512 × 2048
  tokens per step, AdamW with cosine decay).
- `looped/configs/`: model configs. `part1/` and `part2/` hold the release recipes (their learning rates are in
  `recipes.json`), `depth/` holds deeper PLN variants, and `huginn/` holds Huginn-0125.
  `looped/generate_configs.py` writes them.
- `huginn/`:
  - `convert.py`: converts the HF checkpoint `tomg-group-umd/huginn-0125` into this model and checks logit
    parity.
  - `real_exit.py`: token-by-token decoding with real exits.
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
python huginn/real_exit.py --checkpoint checkpoints/huginn-0125/model.pt --check
python huginn/real_exit.py --checkpoint checkpoints/huginn-0125/model.pt --generate --latent-threshold 0.1
```

## Test

```bash
PYTHONPATH=pretrain/OLMo-core/src:. python -m pytest -q looped/tests
```
