#!/bin/bash
"""
EVEREST Ablation Study - Cluster Job Submission Script

This script submits the ablation study array job and optional analysis job
with multiple fallback resource configurations for different cluster systems.

Usage:
  ./submit_jobs.sh [--dry-run] [--components-only] [--sequence-only] [--no-analysis]
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

# Parse command line arguments
DRY_RUN=false
COMPONENTS_ONLY=false
SEQUENCE_ONLY=false
NO_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --components-only)
            COMPONENTS_ONLY=true
            shift
            ;;
        --sequence-only)
            SEQUENCE_ONLY=true
            shift
            ;;
        --no-analysis)
            NO_ANALYSIS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--components-only] [--sequence-only] [--no-analysis]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Show commands without executing"
            echo "  --components-only Run only component ablation (jobs 1-35)"
            echo "  --sequence-only   Run only sequence ablation (jobs 36-60)"
            echo "  --no-analysis     Skip analysis job submission"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p "$LOGS_DIR"

echo "🧪 EVEREST Ablation Study - Cluster Submission"
echo "=============================================="
echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Script directory: $SCRIPT_DIR"
echo "📁 Logs directory: $LOGS_DIR"

if [ "$COMPONENTS_ONLY" = true ]; then
    echo "🔬 Running components ablation only (jobs 1-35)"
elif [ "$SEQUENCE_ONLY" = true ]; then
    echo "📏 Running sequence ablation only (jobs 36-60)"
else
    echo "🔬 Running full ablation study (60 experiments)"
fi

if [ "$NO_ANALYSIS" = true ]; then
    echo "⚠️  Analysis job will be skipped"
fi

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN MODE - No jobs will be submitted"
fi

echo ""

# Determine array range based on options
if [ "$COMPONENTS_ONLY" = true ]; then
    ARRAY_RANGE="1-35"
    EXPERIMENT_COUNT=35
elif [ "$SEQUENCE_ONLY" = true ]; then
    ARRAY_RANGE="36-60"
    EXPERIMENT_COUNT=25
else
    ARRAY_RANGE="1-60"
    EXPERIMENT_COUNT=60
fi

echo "📊 Array job configuration:"
echo "   Range: $ARRAY_RANGE"
echo "   Total experiments: $EXPERIMENT_COUNT"
echo "   Component ablations: 7 variants × 5 seeds = 35 jobs"
echo "   Sequence ablations: 5 lengths × 5 seeds = 25 jobs"
echo "   Resource: 1 GPU, 4 cores, 24GB RAM per job"
echo "   Time limit: 24 hours per job"

# List of PBS scripts to try (in order of preference)
PBS_SCRIPTS=(
    "submit_ablation_array.pbs"
    "submit_ablation_array_minimal.pbs"
)

# Check if this is a SLURM system
if command -v sbatch &> /dev/null; then
    echo ""
    echo "🔍 SLURM detected - using SLURM submission"
    
    SLURM_SCRIPT="$SCRIPT_DIR/submit_ablation_array_slurm.pbs"
    
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "🔍 DRY RUN - SLURM command that would be executed:"
        echo "sbatch --array=$ARRAY_RANGE $SLURM_SCRIPT"
        
        if [ "$NO_ANALYSIS" = false ]; then
            echo "# After array job completes:"
            echo "sbatch --dependency=afterok:<ARRAY_JOB_ID> $SCRIPT_DIR/submit_analysis.pbs"
        fi
    else
        echo ""
        echo "🚀 Submitting SLURM array job..."
        
        # Modify the array range in the SLURM script temporarily
        TEMP_SCRIPT=$(mktemp)
        sed "s/#SBATCH --array=1-60/#SBATCH --array=$ARRAY_RANGE/" "$SLURM_SCRIPT" > "$TEMP_SCRIPT"
        
        ARRAY_JOB_ID=$(sbatch "$TEMP_SCRIPT" | awk '{print $4}')
        rm "$TEMP_SCRIPT"
        
        echo "✅ SLURM array job submitted: $ARRAY_JOB_ID"
        echo "   Range: $ARRAY_RANGE"
        
        # Submit analysis job with dependency
        if [ "$NO_ANALYSIS" = false ]; then
            echo ""
            echo "📊 Submitting SLURM analysis job..."
            
            ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_analysis.pbs"
            
            if ANALYSIS_JOB_ID=$(sbatch --dependency=afterok:$ARRAY_JOB_ID "$ANALYSIS_SCRIPT" 2>/dev/null | awk '{print $4}'); then
                echo "✅ Analysis job submitted: $ANALYSIS_JOB_ID"
                echo "   Dependency: afterok:$ARRAY_JOB_ID"
            else
                echo "⚠️  Could not submit analysis job with dependency"
                echo "   Please submit manually after array job completes:"
                echo "   sbatch $ANALYSIS_SCRIPT"
            fi
        fi
    fi

elif command -v qsub &> /dev/null; then
    echo ""
    echo "🔍 PBS detected - trying PBS submission with fallback configurations"
    
    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "🔍 DRY RUN - PBS commands that would be tried:"
        for script in "${PBS_SCRIPTS[@]}"; do
            echo "qsub -J $ARRAY_RANGE $SCRIPT_DIR/$script"
        done
        
        if [ "$NO_ANALYSIS" = false ]; then
            echo "# After array job completes:"
            echo "qsub -W depend=afterokarray:<ARRAY_JOB_ID> $SCRIPT_DIR/submit_analysis.pbs"
        fi
    else
        echo ""
        echo "🚀 Submitting PBS array job..."
        
        ARRAY_JOB_ID=""
        SUCCESSFUL_SCRIPT=""
        
        # Try each PBS script configuration
        for script in "${PBS_SCRIPTS[@]}"; do
            SCRIPT_PATH="$SCRIPT_DIR/$script"
            
            if [ ! -f "$SCRIPT_PATH" ]; then
                echo "⚠️  Script not found: $script"
                continue
            fi
            
            echo "🔄 Trying configuration: $script"
            
            # Modify the array range in the PBS script temporarily
            TEMP_SCRIPT=$(mktemp)
            sed "s/#PBS -J 1-60/#PBS -J $ARRAY_RANGE/" "$SCRIPT_PATH" > "$TEMP_SCRIPT"
            
            # Try to submit the job
            if ARRAY_JOB_ID=$(qsub "$TEMP_SCRIPT" 2>/dev/null | cut -d'.' -f1); then
                rm "$TEMP_SCRIPT"
                SUCCESSFUL_SCRIPT="$script"
                echo "✅ PBS array job submitted: $ARRAY_JOB_ID"
                echo "   Configuration: $script"
                echo "   Range: $ARRAY_RANGE"
                break
            else
                rm "$TEMP_SCRIPT"
                echo "❌ Failed with configuration: $script"
            fi
        done
        
        if [ -z "$ARRAY_JOB_ID" ]; then
            echo ""
            echo "❌ All PBS configurations failed!"
            echo ""
            echo "🔧 Troubleshooting suggestions:"
            echo "1. Check cluster documentation for correct resource syntax:"
            echo "   https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/queues/job-sizing-guidance/"
            echo ""
            echo "2. Try manual submission with cluster-specific syntax:"
            echo "   qsub -l select=1:ncpus=2:mem=16gb:ngpus=1 -J $ARRAY_RANGE $SCRIPT_DIR/submit_ablation_array.pbs"
            echo ""
            echo "3. Check available partitions/queues:"
            echo "   qstat -Q"
            echo ""
            echo "4. Check your account limits:"
            echo "   qstat -u $USER"
            echo ""
            exit 1
        fi
        
        # Submit analysis job with dependency
        if [ "$NO_ANALYSIS" = false ]; then
            echo ""
            echo "📊 Submitting PBS analysis job..."
            
            ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_analysis.pbs"
            
            # Try different dependency syntaxes for different PBS systems
            if ANALYSIS_JOB_ID=$(qsub -W depend=afterokarray:$ARRAY_JOB_ID "$ANALYSIS_SCRIPT" 2>/dev/null | cut -d'.' -f1); then
                echo "✅ Analysis job submitted: $ANALYSIS_JOB_ID"
                echo "   Dependency: afterokarray:$ARRAY_JOB_ID"
            elif ANALYSIS_JOB_ID=$(qsub -W depend=afterany:$ARRAY_JOB_ID "$ANALYSIS_SCRIPT" 2>/dev/null | cut -d'.' -f1); then
                echo "✅ Analysis job submitted: $ANALYSIS_JOB_ID"
                echo "   Dependency: afterany:$ARRAY_JOB_ID"
            else
                echo "⚠️  Could not submit analysis job with dependency"
                echo "   Please submit manually after array job completes:"
                echo "   qsub $ANALYSIS_SCRIPT"
                ANALYSIS_JOB_ID=""
            fi
        fi
    fi

else
    echo ""
    echo "❌ Neither SLURM (sbatch) nor PBS (qsub) found!"
    echo "   Please install a job scheduler or run locally with:"
    echo "   python models/ablation/run_ablation_study.py --mode all"
    exit 1
fi

echo ""
echo "📋 Job Summary:"
echo "=============="

if [ "$DRY_RUN" = false ]; then
    if command -v sbatch &> /dev/null; then
        echo "Scheduler: SLURM"
        echo "Array Job ID: $ARRAY_JOB_ID"
    else
        echo "Scheduler: PBS"
        echo "Array Job ID: $ARRAY_JOB_ID"
        echo "Configuration: $SUCCESSFUL_SCRIPT"
    fi
    
    if [ "$NO_ANALYSIS" = false ]; then
        if [ -n "$ANALYSIS_JOB_ID" ]; then
            echo "Analysis Job ID: $ANALYSIS_JOB_ID"
        else
            echo "Analysis Job: Manual submission required"
        fi
    fi
else
    echo "DRY RUN - No jobs submitted"
fi

echo ""
echo "📊 Monitoring commands:"
if command -v sbatch &> /dev/null; then
    echo "   squeue -u $USER"
    echo "   squeue -j $ARRAY_JOB_ID  # Array job details"
else
    echo "   qstat -u $USER"
    echo "   qstat -t $ARRAY_JOB_ID  # Array job details"
fi

echo ""
echo "📁 Results will be saved to:"
echo "   models/ablation/results/"
echo "   models/ablation/trained_models/"
echo ""
echo "🎯 Expected completion time: ~24 hours (parallel execution)"
echo "💾 Expected storage usage: ~30GB"

if [ "$COMPONENTS_ONLY" = true ] || [ "$SEQUENCE_ONLY" = true ]; then
    echo ""
    echo "⚠️  Note: Running partial ablation study"
    if [ "$COMPONENTS_ONLY" = true ]; then
        echo "   Only component ablations will be executed"
    else
        echo "   Only sequence length ablations will be executed"
    fi
fi 