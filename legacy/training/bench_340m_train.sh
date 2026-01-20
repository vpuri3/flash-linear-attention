#!/usr/bin/env bash
source ~/.bash_profile
source /project/community/${whoami}/flash-linear-attention/.venv/bin/activate

set -euo pipefail

#================================================================================#
# Train 340M variants for GLA, Transformer++, Mamba, and Mamba2.
# Configure dataset/cache and (optionally) model configs via env vars, then run:
#   DATA=HuggingFaceFW/fineweb-edu \
#   NAME=sample-10BT \
#   CACHE=data/HuggingFaceFW/fineweb-edu/sample-10BT/train \
#   bash legacy/training/bench_340m_train.sh
#   
#================================================================================#
GPUS=4
#================================================================================#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_DIR="$ROOT_DIR/legacy/training"

LR="${LR:-3e-4}"
SCHEDULER="${SCHEDULER:-cosine_with_min_lr}"
BATCH="${BATCH:-32}"
UPDATE="${UPDATE:-1}"
WARMUP="${WARMUP:-1024}"
STEPS="${STEPS:-40960}"
CONTEXT="${CONTEXT:-2048}"
GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
PROJECT="${PROJECT:-fla}"
DATA="${DATA:-HuggingFaceFW/fineweb-edu}"
NAME="${NAME:-sample-10BT}"
WORKERS="${WORKERS:-24}"
PREFETCH="${PREFETCH:-4}"
LOGGING="${LOGGING:-1}"
CACHE="${CACHE:-data/HuggingFaceFW/fineweb-edu/sample-10BT/train}"

GLA_CONFIG="${GLA_CONFIG:-configs/gla_340M.json}"
TRANSFORMER_PP_CONFIG="${TRANSFORMER_PP_CONFIG:-configs/transformer_340M.json}"
MAMBA_CONFIG="${MAMBA_CONFIG:-}"
MAMBA2_CONFIG="${MAMBA2_CONFIG:-}"

run_train() {
  local model_name="$1"
  local model_type="$2"
  local model_config="$3"
  local out_dir="$4"

  if [[ -z "$model_config" ]]; then
    echo "Skipping $model_name (model config not set)."
    return 0
  fi

  (
    cd "$TRAIN_DIR"
    bash ./train.sh \
      type="$model_type" \
      lr="$LR" \
      scheduler="$SCHEDULER" \
      batch="$BATCH" \
      update="$UPDATE" \
      warmup="$WARMUP" \
      steps="$STEPS" \
      context="$CONTEXT" \
      gpus="$GPUS" \
      nodes="$NODES" \
      path="$out_dir" \
      project="$PROJECT" \
      model="$model_config" \
      data="$DATA" \
      name="$NAME" \
      workers="$WORKERS" \
      prefetch="$PREFETCH" \
      logging="$LOGGING" \
      cache="$CACHE"
  )
}

# run_train <<MODEL_NAME>> <<MODEL_TYPE>> <<MODEL_CONFIG>> <<OUTPUT_DIR>>

run_train "gla_340m" "gla" "$GLA_CONFIG" "exp/gla-340m-10B"
run_train "mamba_340m" "mamba" "$MAMBA_CONFIG" "exp/mamba-340m-10B"
run_train "mamba2_340m" "mamba2" "$MAMBA2_CONFIG" "exp/mamba2-340m-10B"
run_train "transformerpp_340m" "transformer" "$TRANSFORMER_PP_CONFIG" "exp/transformer-pp-340m-10B"
#