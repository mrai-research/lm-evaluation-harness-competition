#!/bin/bash
#SBATCH --job-name=lm_eval_dense
#SBATCH --error=logs/lm_eval_%A_%a.err
#SBATCH --output=logs/lm_eval_%A_%a.out
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --array=0-26
# SBATCH --partition=a5000

TASK=mmlu_sfc_basic
PARAMS=1b
ARCH=1

source ~/.bashrc
conda activate e2lm

# Gather all subfolders
SUBFOLDERS=(/hadatasets/morai/dense-$PARAMS-arch$ARCH/*/)
FOLDER=${SUBFOLDERS[$SLURM_ARRAY_TASK_ID]}

accelerate launch -m lm_eval --model hf \
        --model_args pretrained=$FOLDER,dtype=bfloat16 \
        --tasks $TASK \
        --batch_size 32 \
        --log_samples \
        --output_path results/$TASK
