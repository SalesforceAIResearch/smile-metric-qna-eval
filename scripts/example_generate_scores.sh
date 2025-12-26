#!/bin/bash
# =============================================================================
# Score Generation Script (Example)
# =============================================================================
# This script demonstrates how to generate SMILE and baseline metric scores.
# Modify the parameters below for your specific use case.
# =============================================================================

# Set project root (modify as needed)
export PYTHONPATH=./
cd ./

# =============================================================================
# Configuration
# =============================================================================

# Dataset configuration
export dataset="hotpotqa"                           # Dataset name (e.g., hotpotqa, mrqa, docvqa, textvqa, etc.)
export pred_model="gpt4o"                           # Prediction model name
export syn_ans_model="llama-3.2-3b-instruct"        # Synthetic answer generation model
export emb_model="ember-v1"                         # Embedding model for SMILE
export dataset_size="subset_200"                    # Dataset size (subset_200 or full_set)

# Input/Output paths
export input_file="./datasets/${dataset_size}/syn_ans/syn_model-${syn_ans_model}/${dataset}_${pred_model}_data.jsonl"
export output_dir="./evaluations/${dataset_size}/syn_model-${syn_ans_model}/emb_model-${emb_model}/${dataset}/${pred_model}"

# Optional: Save/load embeddings for faster recomputation
# export save_emb_folder="./datasets/embeddings/${dataset_size}/emb_model-${emb_model}/${dataset}/${pred_model}"
# export load_emb_folder="./datasets/embeddings/${dataset_size}/emb_model-${emb_model}/${dataset}/${pred_model}"

# =============================================================================
# Generate All Scores
# =============================================================================

echo "=========================================="
echo "Generating scores for ${dataset}"
echo "Input: ${input_file}"
echo "Output dir: ${output_dir}"
echo "=========================================="

# -----------------------------------------------------------------------------
# SMILE Score
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing SMILE score..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_smile.pkl \
    --emb_model ${emb_model} \
    --syn_ans_model ${syn_ans_model} \
    --eval_mode smile \
    --timeit \
    --verbose

# -----------------------------------------------------------------------------
# ROUGE-L Score
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing ROUGE-L..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_rouge.pkl \
    --eval_mode rouge \
    --timeit

# -----------------------------------------------------------------------------
# BERTScore
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing BERTScore..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_bert_score.pkl \
    --eval_mode bert_score \
    --timeit

# -----------------------------------------------------------------------------
# METEOR
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing METEOR..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_meteor.pkl \
    --eval_mode meteor \
    --timeit

# -----------------------------------------------------------------------------
# Exact Match
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing Exact Match..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_exact_match.pkl \
    --eval_mode exact_match \
    --timeit

# -----------------------------------------------------------------------------
# Sentence-BERT (sBERT)
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing sBERT..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_sbert.pkl \
    --eval_mode sbert \
    --timeit

# -----------------------------------------------------------------------------
# BLEURT
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing BLEURT..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_bleurt.pkl \
    --eval_mode bleurt \
    --timeit

# -----------------------------------------------------------------------------
# MoverScore
# -----------------------------------------------------------------------------
echo ""
echo ">>> Computing MoverScore..."
python3 pyscripts/generate_scores.py \
    --input_file ${input_file} \
    --output_file ${output_dir}/${dataset}_moverscore.pkl \
    --eval_mode moverscore \
    --timeit

echo ""
echo "=========================================="
echo "All scores generated for ${dataset}"
echo "Results saved in: ${output_dir}/"
echo "=========================================="

