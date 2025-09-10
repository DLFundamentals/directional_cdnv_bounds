#!/bin/bash

# --- Configuration ---
GPU_ID=1
LOGS_DIR="logs/"
CKPT_DIR=""
MAX_PARALLEL_JOBS=2

# --- Allow OUTPUT_DIR and CONFIG_FILE to be passed as the first argument, with a default if not provided ---
if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="results/imagenet" # Default value if no argument is provided
fi

if [ -n "$2" ]; then
    CONFIG_FILE="$2"
else
    CONFIG_FILE="configs/simclr_pretrained_imagenet.yaml" # default value
fi

# --- Create output directory if it doesn't exist ---
mkdir -p "${OUTPUT_DIR}"

echo "Starting evaluation for different N_SHOT values..."

# --- Loop through N_SHOT values ---
for SEED in 1 2 3 4 5; do
    echo "Launching evaluation for SEED=${SEED}..."

    # Run the command in the background
    # Redirect stdout and stderr to a unique log file for each run
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u src/nccc_eval.py \
        --config "${CONFIG_FILE}" \
        --ckpt_path "${CKPT_DIR}" \
        --output_path "${OUTPUT_DIR}" \
        --n_shot 1 5 10 20 50 100 200 500 \
        --seed "${SEED}" \
        > "${LOGS_DIR}/nccc_seed_${SEED}.log" 2>&1 &

    # --- Limit parallel jobs ---
    # This waits for a background job to finish if MAX_PARALLEL_JOBS are already running.
    while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
        sleep 5 # Check every 5 seconds
    done
done

# --- Wait for all background jobs to complete ---
echo "All evaluation jobs launched. Waiting for them to finish..."
wait

echo "All evaluation jobs completed."