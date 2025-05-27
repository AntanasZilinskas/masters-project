#!/bin/bash

# Batched submission script for ICL cluster
# This submits ablation jobs in smaller batches to work within queue limits

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🧪 EVEREST Ablation Study - Batched Submission"
echo "=============================================="
echo ""
echo "📁 Project root: $PROJECT_ROOT"

# Parse command line arguments
BATCH_SIZE=10
DRY_RUN=false
START_BATCH=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --start-batch)
            START_BATCH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--batch-size N] [--start-batch N] [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --batch-size N   Number of jobs per batch (default: 10)"
            echo "  --start-batch N  Which batch to start from (default: 1)"
            echo "  --dry-run        Show what would be submitted"
            echo ""
            echo "Examples:"
            echo "  $0                           # Submit first batch of 10 jobs"
            echo "  $0 --batch-size 5            # Submit batches of 5 jobs"
            echo "  $0 --start-batch 2           # Start from batch 2"
            echo "  $0 --dry-run                 # Show what would be done"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Change to project root
cd "$PROJECT_ROOT"

# Calculate total experiments and batches
TOTAL_EXPERIMENTS=60
TOTAL_BATCHES=$(( (TOTAL_EXPERIMENTS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "📊 Batch Configuration:"
echo "   Total experiments: $TOTAL_EXPERIMENTS"
echo "   Batch size: $BATCH_SIZE"
echo "   Total batches: $TOTAL_BATCHES"
echo "   Starting from batch: $START_BATCH"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - Batches that would be submitted:"
    
    for ((batch=$START_BATCH; batch<=TOTAL_BATCHES; batch++)); do
        start_job=$(( (batch - 1) * BATCH_SIZE + 1 ))
        end_job=$(( batch * BATCH_SIZE ))
        if [ $end_job -gt $TOTAL_EXPERIMENTS ]; then
            end_job=$TOTAL_EXPERIMENTS
        fi
        
        echo "   Batch $batch: Jobs $start_job-$end_job"
    done
    
    echo ""
    echo "To submit batch 1:"
    echo "   $0"
    echo ""
    echo "To submit next batch after current completes:"
    echo "   $0 --start-batch 2"
    
    exit 0
fi

# Check current queue status
echo ""
echo "📋 Current queue status:"
CURRENT_JOBS=$(qstat -u $USER 2>/dev/null | grep -c "^[0-9]" || echo "0")
echo "   Your current jobs: $CURRENT_JOBS"

if [ $CURRENT_JOBS -gt 0 ]; then
    echo ""
    echo "⚠️  You have $CURRENT_JOBS jobs in the queue."
    echo "   Consider waiting for some to complete before submitting more."
    echo "   Or use a smaller batch size: --batch-size 5"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 0
    fi
fi

# Submit the specified batch
batch=$START_BATCH
start_job=$(( (batch - 1) * BATCH_SIZE + 1 ))
end_job=$(( batch * BATCH_SIZE ))
if [ $end_job -gt $TOTAL_EXPERIMENTS ]; then
    end_job=$TOTAL_EXPERIMENTS
fi

echo ""
echo "🚀 Submitting Batch $batch (Jobs $start_job-$end_job)..."

# Create a temporary PBS script for this batch
TEMP_PBS=$(mktemp)
cat > "$TEMP_PBS" << EOF
#!/bin/bash
#PBS -N everest_ablation_b${batch}
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1
#PBS -q v1_gpu72
#PBS -J $start_job-$end_job
#PBS -j oe
#PBS -o logs/ablation_\${PBS_ARRAY_INDEX}.log

# EVEREST Ablation Study - Batch $batch
# Jobs $start_job-$end_job of $TOTAL_EXPERIMENTS total

# Load environment
module load anaconda3/personal
source activate everest_env

# Set up paths
cd \$PBS_O_WORKDIR
export PYTHONPATH="\${PBS_O_WORKDIR}:\${PYTHONPATH}"

# Create logs directory
mkdir -p logs

echo "=== EVEREST Ablation Study Job \${PBS_ARRAY_INDEX}/$TOTAL_EXPERIMENTS ==="
echo "Batch: $batch"
echo "Job ID: \$PBS_JOBID"
echo "Array Index: \$PBS_ARRAY_INDEX"
echo "Node: \$(hostname)"
echo "Working Directory: \$(pwd)"
echo "GPU Devices: \$CUDA_VISIBLE_DEVICES"
echo "Start Time: \$(date)"
echo ""

# Define experiment configurations
VARIANTS=("full_model" "no_evidential" "no_evt" "mean_pool" "cross_entropy" "no_precursor" "fp32_training")
SEEDS=(0 1 2 3 4)
SEQ_VARIANTS=("seq_5" "seq_7" "seq_10" "seq_15" "seq_20")

# Calculate which experiment to run based on array index
ARRAY_INDEX=\$((PBS_ARRAY_INDEX - 1))  # Convert to 0-based indexing

if [ \$ARRAY_INDEX -lt 35 ]; then
    # Component ablation experiment
    VARIANT_INDEX=\$((ARRAY_INDEX / 5))
    SEED_INDEX=\$((ARRAY_INDEX % 5))
    
    VARIANT=\${VARIANTS[\$VARIANT_INDEX]}
    SEED=\${SEEDS[\$SEED_INDEX]}
    
    echo "🔬 Running component ablation:"
    echo "   Variant: \$VARIANT"
    echo "   Seed: \$SEED"
    echo ""
    
    python -m ablation.trainer --variant \$VARIANT --seed \$SEED
    
else
    # Sequence length ablation experiment
    SEQ_INDEX=\$(((ARRAY_INDEX - 35) / 5))
    SEED_INDEX=\$(((ARRAY_INDEX - 35) % 5))
    
    SEQ_VARIANT=\${SEQ_VARIANTS[\$SEQ_INDEX]}
    SEED=\${SEEDS[\$SEED_INDEX]}
    
    echo "📏 Running sequence ablation:"
    echo "   Sequence: \$SEQ_VARIANT"
    echo "   Seed: \$SEED"
    echo ""
    
    python -m ablation.trainer --variant full_model --seed \$SEED --sequence \$SEQ_VARIANT
fi

echo ""
echo "✅ Job completed successfully at \$(date)"
EOF

# Submit the batch
SUBMIT_OUTPUT=$(qsub "$TEMP_PBS" 2>&1)
SUBMIT_EXIT_CODE=$?

if [ $SUBMIT_EXIT_CODE -eq 0 ]; then
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | cut -d'.' -f1)
    echo "✅ Batch $batch submitted successfully!"
    echo "   Job ID: $JOB_ID"
    echo "   Jobs: $start_job-$end_job"
    echo "   Full output: $SUBMIT_OUTPUT"
    
    echo ""
    echo "📊 Monitoring commands:"
    echo "   qstat -u $USER"
    echo "   qstat -t $JOB_ID"
    
    echo ""
    echo "🔄 To submit next batch after this completes:"
    next_batch=$((batch + 1))
    if [ $next_batch -le $TOTAL_BATCHES ]; then
        echo "   $0 --start-batch $next_batch"
    else
        echo "   This was the final batch!"
    fi
    
    echo ""
    echo "📁 Results will be saved to:"
    echo "   models/ablation/results/"
    echo "   models/ablation/trained_models/"
    
else
    echo "❌ Batch submission failed!"
    echo "   Exit code: $SUBMIT_EXIT_CODE"
    echo "   Error output: $SUBMIT_OUTPUT"
    exit 1
fi

# Cleanup
rm "$TEMP_PBS" 