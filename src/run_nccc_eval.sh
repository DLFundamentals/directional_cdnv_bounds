#!/bin/bash

# --- Configuration ---
GPU_ID=2
MAX_PARALLEL_JOBS=3

# --- Allow few variables to be passed as argument, with a default if not provided ---
if [ -n "$1" ]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="results/simclr/nccc" # Default value if no argument is provided
fi

if [ -n "$2" ]; then
    CONFIG_FILE="$2"
else
    CONFIG_FILE="configs/simclr_pretrained_imagenet.yaml" # default value
fi

if [ -n "$3" ]; then
    LOGS_DIR="$3"
else
    LOGS_DIR="logs/simclr/nccc" # default value
fi

if [ -n "$4" ]; then
    CKPT_DIR="$4"
else
    CKPT_DIR="" # default value
fi



# --- Create output directory if it doesn't exist ---
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

echo "Starting evaluation for different N_SHOT values..."

# --- Loop through N_SHOT values ---
for SEED in 1 2 3 4 5; do
    echo "Launching evaluation for SEED=${SEED}..."

    for N_SHOT in 1 5 10 20 50 100 200 500; do
        echo "  Running n_shot=${N_SHOT}..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u src/nccc_eval.py \
            --config "${CONFIG_FILE}" \
            --ckpt_path "${CKPT_DIR}" \
            --output_path "${OUTPUT_DIR}" \
            --n_shot "${N_SHOT}" \
            --seed "${SEED}" \
            > "${LOGS_DIR}/nccc_seed_${SEED}_nshot_${N_SHOT}.log" 2>&1 &

        # --- Limit parallel jobs ---
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
            sleep 5 # Check every 5 seconds
        done
    done

done

# --- Wait for all background jobs to complete ---
echo "All evaluation jobs launched. Waiting for them to finish..."
wait

echo "All evaluation jobs completed."