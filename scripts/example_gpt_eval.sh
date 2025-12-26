#!/bin/bash
# =============================================================================
# GPT-based Evaluation Script (Example)
# =============================================================================
# This script demonstrates how to run GPT-based evaluation on QA pairs.
# Modify the parameters below for your specific use case.
# =============================================================================

# Set project root (modify as needed)
export PYTHONPATH=./
cd ./

# =============================================================================
# Configuration
# =============================================================================

# Dataset configuration
export dataset="hotpotqa"                           # Dataset name
export pred_model="gpt4o"                           # Prediction model name
export syn_model="llama-3.2-3b-instruct"            # Synthetic answer model
export emb_model="ember-v1"                         # Embedding model
export dataset_size="subset_200"                    # Dataset size

# OpenAI Configuration
# IMPORTANT: Replace with your actual API key
export api_key="YOUR_OPENAI_API_KEY"
export openai_model="gpt-3.5-turbo"                 # Options: "gpt-3.5-turbo", "gpt-4o"
export num_tasks=8                                   # Number of parallel tasks

# Input/Output paths
export pred_path="./datasets/${dataset_size}/syn_ans/syn_model-${syn_model}/${dataset}_${pred_model}_data.jsonl"
export output_dir="./evaluations/${dataset_size}/syn_model-${syn_model}/emb_model-${emb_model}/${dataset}/${pred_model}/vqa/${openai_model}"
export output_json="./evaluations/${dataset_size}/syn_model-${syn_model}/emb_model-${emb_model}/${dataset}/${pred_model}/${dataset}_${openai_model}_results.json"

# =============================================================================
# Run GPT-based Evaluation
# =============================================================================

echo "Running ${openai_model} evaluation on ${dataset}..."
echo "Input: ${pred_path}"
echo "Output: ${output_json}"

python3 pyscripts/eval_gpt_pref.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --num_tasks ${num_tasks} \
    --openai_model ${openai_model} \
    --dataset ${dataset} \
    --timeit

# Wait for all tasks to complete
wait

# Clean up intermediate files
rm -rf ${output_dir}

echo "Completed ${openai_model} evaluation on ${dataset}"
echo "Results saved at: ${output_json}"

