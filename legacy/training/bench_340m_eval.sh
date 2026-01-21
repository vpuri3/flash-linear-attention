#!/usr/bin/env bash
source ~/.bash_profile
source /project/community/${USER}/flash-linear-attention/.venv/bin/activate

set -euo pipefail

#================================================================================#
# Evaluate 340M checkpoints on MMLU, CSR tasks, WikiText, LAMBADA, and LongBench.
# Configure model paths via env vars before running:
#   GLA_MODEL=/path/to/gla \
#   TRANSFORMER_PP_MODEL=/path/to/transformer_pp \
#   MAMBA_MODEL=/path/to/mamba \
#   MAMBA2_MODEL=/path/to/mamba2 \
#   bash legacy/training/bench_340m_eval.sh
#================================================================================#
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${ROOT_DIR}/legacy/training/exp"
RESULTS_DIR="${ROOT_DIR}/legacy/training/results/bench_340m"

#================================================================================#
GLA_MODEL="${CKPT_DIR}/gla-340m-10B"
MAMBA_MODEL="${CKPT_DIR}/mamba-340m-10B"
MAMBA2_MODEL="${CKPT_DIR}/mamba2-340m-10B"
TRANSFORMER_PP_MODEL="${CKPT_DIR}/transformer-pp-340m-10B"
GATED_TRANSFORMER_MODEL="${CKPT_DIR}/gated-transformer-340m-10B"

#================================================================================#
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-4}"
MMLU_FEWSHOT="${MMLU_FEWSHOT:-5}"
CORE_FEWSHOT="${CORE_FEWSHOT:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
LONGBENCH_BATCH_SIZE="${LONGBENCH_BATCH_SIZE:-2}"
LONGBENCH_FEWSHOT="${LONGBENCH_FEWSHOT:-0}"
LONGBENCH_MAX_LENGTH="${LONGBENCH_MAX_LENGTH:-4096}"

TASKS_CSR="winogrande,piqa,arc_challenge,openbookqa,arc_easy,boolq"
TASKS_WIKI_LMB="wikitext,lambada_openai"
TASKS_MMLU="mmlu"
TASKS_LONGBENCH="${TASKS_LONGBENCH:-longbench_2wikimqa,longbench_hotpotqa,longbench_musique,longbench_qasper,longbench_narrativeqa,longbench_samsum,longbench_triviaqa,longbench_passage_count,longbench_passage_retrieval_zh,longbench_qmsum,longbench_gov_report,longbench_multi_news,longbench_repobench-p,longbench_lcc}"

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
  local longbench_model_args="${model_args},max_length=${LONGBENCH_MAX_LENGTH}"
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

  echo "Running $model_name LongBench..."
  python -m evals.harness \
    --model hf \
    --model_args "$longbench_model_args" \
    --tasks "$TASKS_LONGBENCH" \
    --batch_size "$LONGBENCH_BATCH_SIZE" \
    --num_fewshot "$LONGBENCH_FEWSHOT" \
    --device "$DEVICE" \
    --output_path "$model_dir/longbench.json" \
    --show_config \
    "${limit_args[@]}" \
    2>&1 | tee "$model_dir/longbench.log"
}

#================================================================================#
CUDA_VISIBLE_DEVICES=0 run_eval "gla_340m" "${GLA_MODEL:-}" &
# run_eval "mamba_340m" "${MAMBA_MODEL:-}"
# run_eval "mamba2_340m" "${MAMBA2_MODEL:-}"
CUDA_VISIBLE_DEVICES=1 run_eval "transformerpp_340m" "${TRANSFORMER_PP_MODEL:-}" &
CUDA_VISIBLE_DEVICES=2 run_eval "gated_transformer_340m" "${GATED_TRANSFORMER_MODEL:-}" &
#================================================================================#
#