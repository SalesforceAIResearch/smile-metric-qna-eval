#!/bin/bash
# =============================================================================
# Synthetic Answer Generation Script (Example)
# =============================================================================
# This script demonstrates how to generate synthetic answers using an LLM.
# Modify the parameters below for your specific use case.
# =============================================================================

# conda activate smile_env

# Set project root (modify as needed)
export PYTHONPATH=./
cd ./

# =============================================================================
# Configuration
# =============================================================================

# Dataset configuration
export dataset="hotpotqa"                           # Dataset name
export pred_model="gpt4o"                           # Prediction model used
export dataset_size="subset_200"                    # Dataset size

# Model for synthetic answer generation
# Options: "gpt-3.5-turbo" (API-based) or "meta-llama/Llama-3.2-3B-Instruct" (local)
export model="meta-llama/Llama-3.2-3B-Instruct"

# Input: File with predictions (should contain 'question', 'answer', 'pred' fields)
export input_file="./datasets/${dataset_size}/pred/${dataset}_${pred_model}_data.jsonl"

# Output: File with synthetic answers added (adds 'syn_ans' field)
export output_file="./datasets/${dataset_size}/syn_ans/syn_model-llama-3.2-3b-instruct/${dataset}_${pred_model}_data.jsonl"

# GPU configuration (for local models)
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Run Synthetic Answer Generation
# =============================================================================

echo "Generating synthetic answers for ${dataset}..."
echo "Input: ${input_file}"
echo "Output: ${output_file}"
echo "Model: ${model}"

python3 pyscripts/generate_syn_ans.py \
    --input_file ${input_file} \
    --output_file ${output_file} \
    --model ${model}

echo "Synthetic answer generation completed"
echo "Output saved at: ${output_file}"

