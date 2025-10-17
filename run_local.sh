#!/bin/bash

TASK=mmlu_sfc_basic
PARAMS=1b
ARCH=1

source ~/.bashrc
conda activate e2lm

# Gather all subfolders
SUBFOLDERS=(/hadatasets/morai/dense-$PARAMS-arch$ARCH/*/)
FOLDER=${SUBFOLDERS[0]}

accelerate launch -m lm_eval --model hf \
        --model_args pretrained=$FOLDER,dtype=bfloat16 \
        --tasks $TASK \
        --batch_size 32 \
        --log_samples \
        --output_path results/$TASK
