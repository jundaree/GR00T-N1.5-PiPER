#!/bin/bash

# Base configuration
EMBODIMENT_TAG="new_embodiment"
DATA_CONFIG="piper"
DATASET_PATH="datasets_preprocessed/val_dataset_mujoco"
MODALITY_KEYS="single_arm gripper"
MODEL_BASE_PATH="outputs/gr00t_finetune_mujoco_lora"
OUTPUT_FILE="mse_results.txt"

# Clear previous results file
> ${OUTPUT_FILE}

# Function to format seconds into human-readable time
format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# Start total timer
total_start=$(date +%s)

echo "=========================================="
echo "Evaluating policies across checkpoints"
echo "=========================================="
echo ""
echo "Results will be saved to: ${OUTPUT_FILE}"
echo ""

# Checkpoint array
checkpoints=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
total_checkpoints=${#checkpoints[@]}
current=0

# Loop through checkpoints from 1000 to 10000 with step of 1000
for checkpoint in "${checkpoints[@]}"; do
    ((current++))
    
    # echo "=========================================="
    # echo "Progress: ${current}/${total_checkpoints}"
    # echo "Evaluating checkpoint-${checkpoint}"
    # echo "=========================================="
    
    # Start checkpoint timer
    checkpoint_start=$(date +%s)
    
    # Run evaluation and capture output
    output=$(python scripts/eval_policy.py --trajs 10 \
        --embodiment_tag ${EMBODIMENT_TAG} \
        --model_path ${MODEL_BASE_PATH}/checkpoint-${checkpoint} \
        --data_config ${DATA_CONFIG} \
        --dataset_path ${DATASET_PATH} \
        --modality_keys ${MODALITY_KEYS} 2>&1)
    
    # Calculate checkpoint elapsed time
    checkpoint_end=$(date +%s)
    checkpoint_elapsed=$((checkpoint_end - checkpoint_start))
    
    # Print the output
    echo "$output"
    
    # Extract Average MSE and save to file
    avg_mse=$(echo "$output" | grep "Average MSE across all trajs:" | awk '{print $NF}')
    
    if [ -n "$avg_mse" ]; then
        echo "checkpoint-${checkpoint}: ${avg_mse}" >> ${OUTPUT_FILE}
        echo "✓ Saved MSE: ${avg_mse}"
    else
        echo "✗ Warning: Could not extract MSE for checkpoint-${checkpoint}"
    fi
    
    # Display timing info
    echo ""
    echo "⏱️  Checkpoint time: $(format_time $checkpoint_elapsed)"
    
    # Calculate and display estimated time remaining
    total_elapsed=$((checkpoint_end - total_start))
    avg_time_per_checkpoint=$((total_elapsed / current))
    remaining_checkpoints=$((total_checkpoints - current))
    estimated_remaining=$((avg_time_per_checkpoint * remaining_checkpoints))
    
    if [ $remaining_checkpoints -gt 0 ]; then
        echo "⏱️  Elapsed total: $(format_time $total_elapsed)"
        echo "⏱️  Estimated remaining: $(format_time $estimated_remaining)"
        echo "⏱️  Estimated completion: $(date -d "@$((checkpoint_end + estimated_remaining))" '+%Y-%m-%d %H:%M:%S')"
    fi
    
    echo ""
done

# Calculate total time
total_end=$(date +%s)
total_elapsed=$((total_end - total_start))

echo "=========================================="
echo "All checkpoints evaluated!"
echo "⏱️  Total time: $(format_time $total_elapsed)"
echo "=========================================="
echo ""
echo "Summary of results:"
cat ${OUTPUT_FILE}