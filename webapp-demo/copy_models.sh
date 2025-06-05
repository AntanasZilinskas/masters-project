#!/bin/bash

# Script to copy the newest model versions from remote cluster
# Remote path: /rds/general/user/az2221/home/repositories/masters-project/models/models
# Local path: /Users/antanaszilinskas/Github/master-everest-webapp/Solar-Flare-Prediction-System/models/models

REMOTE_USER="az2221"
REMOTE_HOST="login.hpc.ic.ac.uk"
REMOTE_PATH="/rds/general/user/az2221/home/repositories/masters-project/models/models"
LOCAL_PATH="/Users/antanaszilinskas/Github/master-everest-webapp/Solar-Flare-Prediction-System/models/models"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"

echo "Starting model copy from remote cluster..."
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "Local: ${LOCAL_PATH}"
echo ""

# Define the newest models to copy based on the provided list
# Selecting the highest version available for each model type and horizon
MODELS_TO_COPY=(
    # C-class models (v1.3 is newest available)
    "EVEREST-v1.3-C-24h"
    "EVEREST-v1.3-C-48h" 
    "EVEREST-v1.3-C-72h"
    
    # M-class models 
    "EVEREST-v1.3-M-24h"    # v1.3 available
    "EVEREST-v1.3-M-48h"    # v1.3 available  
    "EVEREST-v1.2-M-72h"    # v1.3 not available, v1.2 is newest
    
    # M5-class models (v1.3 is newest available for all)
    "EVEREST-v1.3-M5-24h"
    "EVEREST-v1.3-M5-48h"
    "EVEREST-v1.3-M5-72h"
)

# Function to copy a single model
copy_model() {
    local model_name=$1
    echo "Copying ${model_name}..."
    
    # Use rsync for efficient copying with progress
    rsync -avz --progress \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/${model_name}/" \
        "${LOCAL_PATH}/${model_name}/"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully copied ${model_name}"
    else
        echo "❌ Failed to copy ${model_name}"
        return 1
    fi
    echo ""
}

# Copy all models
failed_models=()
successful_models=()

for model in "${MODELS_TO_COPY[@]}"; do
    if copy_model "$model"; then
        successful_models+=("$model")
    else
        failed_models+=("$model")
    fi
done

# Summary
echo "========================================="
echo "COPY SUMMARY"
echo "========================================="
echo "Successfully copied ${#successful_models[@]} models:"
for model in "${successful_models[@]}"; do
    echo "  ✅ $model"
done

if [ ${#failed_models[@]} -gt 0 ]; then
    echo ""
    echo "Failed to copy ${#failed_models[@]} models:"
    for model in "${failed_models[@]}"; do
        echo "  ❌ $model"
    done
fi

echo ""
echo "Models copied to: ${LOCAL_PATH}"
echo "Done!" 