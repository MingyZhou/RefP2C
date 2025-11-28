#!/bin/bash
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-key-here}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-your-url-here}"

set -e

# Phase 1: Initial Code Implementation
# Phase 2: Supervisory Signal Design
# Phase 3: Iterative Code Reflection


# =============================
# RefP2C Configuration
# =============================
PAPER_ID="2401.01967"          # The paper identifier (folder under paper/)
WORKSPACE_DIR="test_run_new"   # Directory to store all intermediate and final outputs

# --- Model Configuration ---
MODEL_PHASE_1="gpt-4o-mini"    # Model for Initial Code Implementation
MODEL_PHASE_2="gpt-4o-mini"    # Model for Supervisory Signal Design
MODEL_PHASE_2_RERANK="gpt-4o-mini"    # Model for Retriveval Reranking
MODEL_PHASE_3_EVAL="gpt-4o-mini"    # Model for evaluating generated code
MODEL_PHASE_3_PLAN="gpt-4o-mini"    # Model for planning code revisions
MODEL_PHASE_3_REVISE="gpt-4o-mini"   # Model for revising code
MAX_ITERATIONS=1    # Number of reflection cycles to perform
# REPLACE_FLAG="--replace"     # Optional: overwrite previous results

LOG_DIR="${WORKSPACE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# --- Main Orchestration Logic ---
echo "================================================================="
echo "🚀 STARTING RefP2C for Paper: ${PAPER_ID}"
echo "================================================================="

echo -e "\nPHASE 1: Launching Code Generation..."

python -m scripts.generate_initial_code \
    --paper_id "${PAPER_ID}" \
    --workspace_dir "${WORKSPACE_DIR}" \
    --model "${MODEL_PHASE_1}" \
    2>&1 | tee "${LOG_DIR}/phase1_generate.log"

echo -e "\nPHASE 2: Launching Signal Design..."

python -m scripts.design_signals \
    --paper_id "${PAPER_ID}" \
    --workspace_dir "${WORKSPACE_DIR}" \
    --model "${MODEL_PHASE_2}" \
    --rerank_model "${MODEL_PHASE_2_RERANK}" \
    2>&1 | tee "${LOG_DIR}/phase2_signals.log"

echo "✅ Code Generation and Signal Design completed successfully."

echo -e "\nPHASE 3: Launching Code Reflection Pipeline..."

python -m scripts.reflect_code \
    --paper_id "${PAPER_ID}" \
    --workspace_dir "${WORKSPACE_DIR}" \
    --model_eval "${MODEL_PHASE_3_EVAL}" \
    --model_plan "${MODEL_PHASE_3_PLAN}" \
    --model_revise "${MODEL_PHASE_3_REVISE}" \
    --max_attempts "${MAX_ITERATIONS}" \
    2>&1 | tee "${LOG_DIR}/phase3_reflect.log"

echo -e "\n================================================================="
echo "🎉🎉🎉 COMPLETED SUCCESSFULLY! 🎉🎉🎉"
echo "================================================================="