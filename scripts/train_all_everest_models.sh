#!/bin/bash
# This script submits jobs to train EVEREST models for all combinations of:
# - Flare classes: C, M, M5
# - Time windows: 24h, 48h, 72h

# Create logs directory if it doesn't exist
mkdir -p $HOME/projects/everest/logs

# Store job IDs for tracking
JOB_IDS=()
JOB_CONFIGS=()

# Function to submit a job for a specific configuration
submit_training_job() {
    local flare_class=$1
    local time_window=$2
    
    echo "Submitting job for ${flare_class} flares with ${time_window}h window..."
    
    # Set environment variables for the PBS script
    export FLARE_CLASS=$flare_class
    export TIME_WINDOW=$time_window
    
    # Submit the job and capture the job ID
    job_id=$(qsub -v FLARE_CLASS=$flare_class,TIME_WINDOW=$time_window $HOME/projects/everest/scripts/everest_train.pbs)
    
    # Store the job ID and configuration
    JOB_IDS+=("$job_id")
    JOB_CONFIGS+=("${flare_class}-${time_window}h")
    
    echo "Submitted job ${job_id} for ${flare_class} flares with ${time_window}h window"
    
    # Wait a few seconds between submissions to avoid overwhelming the queue
    sleep 5
}

echo "=== Training EVEREST Models for All Configurations ==="
echo "Starting job submissions at $(date)"

# Submit jobs for all configurations
for flare_class in "C" "M" "M5"; do
    for time_window in "24" "48" "72"; do
        submit_training_job $flare_class $time_window
    done
done

# Print a summary of submitted jobs
echo ""
echo "=== Job Submission Summary ==="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
for i in "${!JOB_IDS[@]}"; do
    echo "Job ${JOB_IDS[$i]}: ${JOB_CONFIGS[$i]}"
done

echo ""
echo "You can monitor the jobs with:"
echo "qstat -u $USER"
echo ""
echo "To check specific job details:"
echo "qstat -f <job_id>"
echo ""
echo "Training logs will be available in:"
echo "$HOME/projects/everest/logs/" 