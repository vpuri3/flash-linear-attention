#!/usr/bin/env bash
set -euo pipefail

source ~/.bash_profile
source /project/community/${USER}/flash-linear-attention/.venv/bin/activate

#================================================================================#
# Train 340M variants for GLA, Transformer++, Mamba, and Mamba2.
# Configure dataset/cache and (optionally) model configs via env vars, then run:
#   DATA=HuggingFaceFW/fineweb-edu \
#   NAME=sample-10BT \
#   CACHE=data/HuggingFaceFW/fineweb-edu/sample-10BT/train \
#   bash legacy/training/bench_340m_train.sh
#   
#================================================================================#
# Training configuration:
CONTEXT=2048  # Context length
BATCH=32      # Batch size
GPUS=4        # Number of GPUs
NODES=1       # Number of nodes
STEPS=40960   # Number of steps
TARGET_TOKENS=$((2048 * 32 * 4 * 40960)) # 10.7B

#================================================================================#
# DEBUGGING (comment this out when done)
#================================================================================#
# GPUS=4
# STEPS=40960
# LR=1e-3
# WANDB_DISABLED=true
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
PREFETCH="${PREFETCH:-8}"
LOGGING="${LOGGING:-1}"
CACHE="${CACHE:-data/HuggingFaceFW/fineweb-edu/sample-10BT/train}"
WANDB_DISABLED="${WANDB_DISABLED:-false}"

GLA_CONFIG="${GLA_CONFIG:-configs/gla_340M.json}"
MAMBA_CONFIG="${MAMBA_CONFIG:-configs/mamba_340m.json}"
MAMBA2_CONFIG="${MAMBA2_CONFIG:-configs/mamba2_340m.json}"
TRANSFORMER_PP_CONFIG="${TRANSFORMER_PP_CONFIG:-configs/transformer_340M.json}"
GATED_TRANSFORMER_CONFIG="${GATED_TRANSFORMER_CONFIG:-configs/gated_transformer_340M.json}"

run_train() {
  local model_name="$1"
  local model_type="$2"
  local model_config="$3"
  local out_dir="$4"
  shift 4  # Remove the first 4 arguments, leaving any additional ones in $@

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
      cache="$CACHE" \
      wandb_disabled="$WANDB_DISABLED" \
      "$@"  # Pass additional args (will override defaults)
  )
}

#================================================================================#
# run_train <<MODEL_NAME>> <<MODEL_TYPE>> <<MODEL_CONFIG>> <<OUTPUT_DIR>> [ADDITIONAL_ARGS...]
# EX: run_train "gla_340m" "gla" "$GLA_CONFIG" "exp/gla-340m-10B" lr=1e-3 checkpoint=exp/gla-340m-10B/checkpoint-8192

# run_train "gla_340m" "gla" "$GLA_CONFIG" "exp/gla-340m-10B"
# run_train "transformerpp_340m" "transformer" "$TRANSFORMER_PP_CONFIG" "exp/transformer-pp-340m-10B"

###
### mamba
###
SCALE=1
KW="batch=$(($BATCH / $SCALE)) update=$(($UPDATE * $SCALE)) steps=$(($STEPS * $SCALE))"
run_train "mamba_340m" "mamba" "$MAMBA_CONFIG" "exp/mamba-340m-10B" ${KW} checkpoint=exp/mamba-340m-10B/checkpoint-34816
run_train "mamba2_340m" "mamba2" "$MAMBA2_CONFIG" "exp/mamba2-340m-10B" ${KW}

###
### testing
###

# run_train "gated_transformer_340m" "gated_transformer" \
#   "$GATED_TRANSFORMER_CONFIG" "exp/gated-transformer-340m-10B" \
#   lr=1e-3 checkpoint=exp/gated-transformer-340m-10B/checkpoint-6144

# run_train "gla_340m" "gla" "$GLA_CONFIG" "exp/dump"
# run_train "mamba_340m" "mamba" "$MAMBA_CONFIG" "exp/mamba-340m-10B"
#================================================================================#
#