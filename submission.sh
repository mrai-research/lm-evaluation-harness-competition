#!/bin/bash

# This script automates the creation of a submission package based on the
# instructions from the provided notebook. It creates a 'submission.zip'
# file containing an 'evaluation.patch' and a 'metadata.yaml'.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Validation ---
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <task_name> <task_metric> <hf_token>"
    echo "  <task_name>: The name of your task (e.g., 'my_benchmark')."
    echo "  <task_metric>: The metric for scoring (e.g., 'acc,norm')."
    echo "  <hf_token>: Your Hugging Face token (use '' for an empty token)."
    exit 1
fi

TASK_NAME=$1
TASK_METRIC=$2
HF_TOKEN=$3
REPO_DIR="lm-evaluation-harness-competition"
SUBMISSION_DIR="submission"

# --- Repository Setup ---
echo "Cloning competition repository..."
cd ../
mkdir tmp_lm_eval
cd tmp_lm_eval
git clone https://github.com/tiiuae/lm-evaluation-harness-competition

cd "$REPO_DIR"
rm -rf *
cd ../../"$REPO_DIR"
cp -r * ../tmp_lm_eval/"$REPO_DIR"/




# --- Prepare Patch File ---
# This script assumes you have already placed your custom benchmark files
# inside the repository (e.g., in 'lm_eval/tasks/').

echo "Staging all new and modified files..."
cd ../tmp_lm_eval/"$REPO_DIR"
git add .

# Verify that there are changes to commit.
if git diff --cached --quiet; then
    echo "Error: No changes have been staged. Add your benchmark files to the repository before running this script."
    exit 1
fi

echo "Creating submission directory..."
mkdir -p "../$SUBMISSION_DIR"

echo "Generating evaluation.patch from staged changes..."
git diff --cached > "../$SUBMISSION_DIR/evaluation.patch"

# --- Prepare Metadata File ---
echo "Creating metadata.yaml..."
cat << EOF > "../$SUBMISSION_DIR/metadata.yaml"
task_name: ${TASK_NAME}
task_metric: ${TASK_METRIC}
hf_token: '${HF_TOKEN}'
EOF

# --- Create Final Zip Archive ---
echo "Creating final submission.zip..."
cd "../$SUBMISSION_DIR"
# The zip command packages the contents of the current directory.
zip -r ../submission.zip ./*
mv ../submission.zip ./
cd ..
cp -r "$SUBMISSION_DIR" ../"$REPO_DIR"/



echo "Process complete. 'submission.zip' has been created in the current directory."