#!/usr/bin/env bash
# Trains one model with torchrun on every GPU of this node.
#
#   launch.sh [--microbatch N] [--hard-stop STEP] MODEL_CONFIG SCALE LEARNING_RATE [RESUME_RUN_DIR]
#
# MODEL_CONFIG is a model JSON, for example looped/configs/part2/s_fixed_pln5.json. SCALE names the
# training recipe in pretrain/configs/SCALE/train.json (400m, 1p5b or ablation). LEARNING_RATE is the
# peak learning rate (looped/configs/recipes.json lists the release's). The global batch is fixed; the
# GPU count and the microbatch (default 16 sequences) only change gradient accumulation. --hard-stop
# ends the run at STEP while the schedule still runs to the recipe's max_steps. With RESUME_RUN_DIR the
# run continues from its latest checkpoint.
#
# Environment: DATA_DIR (tokenized FineWeb-Edu, required), WANDB_ENTITY (required), WANDB_PROJECT
# (default looped), CHECKPOINT_DIR (default ./checkpoints), SCRATCH_DIR for compiler caches (default
# $TMPDIR or /tmp), RUN_ID to end the run name (default the launch time).
set -Eeuo pipefail
usage="usage: $0 [--microbatch N] [--hard-stop STEP] MODEL_CONFIG SCALE LEARNING_RATE [RESUME_RUN_DIR]"
microbatch=16 hard_stop=()
while [[ "${1:-}" == --* ]]; do
  [[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }
  case $1 in
    --microbatch) microbatch=$2 ;;
    --hard-stop) hard_stop=(--hard-stop-step "$2") ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
  shift 2
done
[[ $# -ge 3 && $# -le 4 ]] || { echo "$usage" >&2; exit 2; }
model_config=$(realpath "$1") scale=$2 learning_rate=$3
[[ $microbatch =~ ^[1-9][0-9]*$ ]] || { echo "the microbatch must be a positive integer" >&2; exit 2; }
: "${DATA_DIR:?set DATA_DIR to the tokenized FineWeb-Edu directory}"
: "${WANDB_ENTITY:?set WANDB_ENTITY to your W&B entity}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
train_config="$repo_root/pretrain/configs/$scale/train.json"
[[ -f $model_config ]] || { echo "no model config $model_config" >&2; exit 2; }
[[ -f $train_config ]] || { echo "no training recipe $train_config" >&2; exit 2; }

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
# The learning rate appears in the run name as 0p001 for 1e-3.
lr_tag=$(LC_ALL=C printf '%.10f' "$learning_rate" | sed 's/0*$//; s/\./p/')
model=$(basename "$model_config" .json)
run_name="$model-$scale-lr$lr_tag-s42-$run_id"
save_folder="${CHECKPOINT_DIR:-$repo_root/checkpoints}/$run_name"
resume_args=()
if [[ -n ${4:-} ]]; then
  [[ -d $4 ]] || { echo "resume directory does not exist: $4" >&2; exit 3; }
  save_folder=$(cd "$4" && pwd)
  resume_args=(--resume-run-dir "$save_folder")
else
  [[ ! -e $save_folder ]] || { echo "output already exists: $save_folder" >&2; exit 3; }
fi

runtime_root="${SCRATCH_DIR:-${TMPDIR:-/tmp}}/$run_name"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor"
export TRITON_CACHE_DIR="$runtime_root/triton"
export WANDB_DIR="$runtime_root/wandb"
export PYTHONUNBUFFERED=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR"
cd "$repo_root"

exec torchrun --standalone --nproc-per-node=gpu pretrain/train.py \
  --model-config "$model_config" --train-config "$train_config" \
  --learning-rate "$learning_rate" --device-microbatch-size "$microbatch" "${hard_stop[@]}" \
  --data-dir "$DATA_DIR" --run-name "$run_name" --save-folder "$save_folder" --work-dir "$runtime_root" \
  "${resume_args[@]}" \
  --wandb-project "${WANDB_PROJECT:-looped}" --wandb-entity "$WANDB_ENTITY"
