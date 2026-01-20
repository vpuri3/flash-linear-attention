#!/usr/bin/env bash
source ~/.bash_profile
source /project/community/${whoami}/flash-linear-attention/.venv/bin/activate

set -euo pipefail

#================================================================================#
# Evaluate 340M checkpoints on MMLU, CSR tasks, WikiText, and LAMBADA.
# Configure model paths via env vars before running:
#   GLA_MODEL=/path/to/gla \
#   TRANSFORMER_PP_MODEL=/path/to/transformer_pp \
#   MAMBA_MODEL=/path/to/mamba \
#   MAMBA2_MODEL=/path/to/mamba2 \
#   bash legacy/training/benchmarks_340m.sh
#================================================================================#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/evals/results/bench_340m}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-4}"
MMLU_FEWSHOT="${MMLU_FEWSHOT:-5}"
CORE_FEWSHOT="${CORE_FEWSHOT:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

TASKS_CSR="winogrande,piqa,arc_challenge,openbookqa,arc_easy,boolq"
TASKS_WIKI_LMB="wikitext,lambada_openai"
TASKS_MMLU="mmlu"

mkdir -p "$RESULTS_DIR"

run_eval() {
  local model_name="$1"
  local model_path="$2"

  if [[ -z "$model_path" ]]; then
    echo "Skipping $model_name (model path not set)."
    return 0
  fi

  local model_tag="${MODEL_TAG:-$(basename "$model_path")}"
  local model_dir="$RESULTS_DIR/$model_name/$model_tag"
  mkdir -p "$model_dir"

  local model_args="pretrained=$model_path,dtype=$DTYPE,trust_remote_code=True"
  local limit_args=()
  if [[ -n "$EVAL_LIMIT" ]]; then
    limit_args+=(--limit "$EVAL_LIMIT")
  fi

  echo "Running $model_name MMLU..."
  python -m evals.harness \
    --model hf \
    --model_args "$model_args" \
    --tasks "$TASKS_MMLU" \
    --batch_size "$MMLU_BATCH_SIZE" \
    --num_fewshot "$MMLU_FEWSHOT" \
    --device "$DEVICE" \
    --output_path "$model_dir/mmlu.json" \
    --show_config \
    "${limit_args[@]}" \
    2>&1 | tee "$model_dir/mmlu.log"

  echo "Running $model_name CSR tasks..."
  python -m evals.harness \
    --model hf \
    --model_args "$model_args" \
    --tasks "$TASKS_CSR" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$CORE_FEWSHOT" \
    --device "$DEVICE" \
    --output_path "$model_dir/csr.json" \
    --show_config \
    "${limit_args[@]}" \
    2>&1 | tee "$model_dir/csr.log"

  echo "Running $model_name Wiki/LMB..."
  python -m evals.harness \
    --model hf \
    --model_args "$model_args" \
    --tasks "$TASKS_WIKI_LMB" \
    --batch_size "$BATCH_SIZE" \
    --num_fewshot "$CORE_FEWSHOT" \
    --device "$DEVICE" \
    --output_path "$model_dir/wiki_lmb.json" \
    --show_config \
    "${limit_args[@]}" \
    2>&1 | tee "$model_dir/wiki_lmb.log"
}

run_eval "gla_340m" "${GLA_MODEL:-}"
# run_eval "transformer_pp_340m" "${TRANSFORMER_PP_MODEL:-}"
# run_eval "mamba_340m" "${MAMBA_MODEL:-}"
# run_eval "mamba2_340m" "${MAMBA2_MODEL:-}"
#