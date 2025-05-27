#!/bin/bash
"""
EVEREST Ablation Study - Cluster Job Submission Script

This script submits the complete ablation study workflow:
1. Array job for all ablation experiments (60 jobs)
2. Analysis job that runs after all experiments complete

Usage:
  ./submit_jobs.sh [--dry-run] [--component-only] [--sequence-only]
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

# Parse command line arguments
DRY_RUN=false
COMPONENT_ONLY=false
SEQUENCE_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --component-only)
            COMPONENT_ONLY=true
            shift
            ;;
        --sequence-only)
            SEQUENCE_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--component-only] [--sequence-only]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Show commands without executing"
            echo "  --component-only  Run only component ablations (35 jobs)"
            echo "  --sequence-only   Run only sequence length ablations (25 jobs)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create necessary directories
echo "🔧 Setting up directories..."
mkdir -p "$LOGS_DIR"
mkdir -p "$PROJECT_ROOT/models/ablation/results"
mkdir -p "$PROJECT_ROOT/models/ablation/plots"
mkdir -p "$PROJECT_ROOT/models/ablation/logs"

# Change to project root
cd "$PROJECT_ROOT"

echo "🔬 EVEREST Ablation Study - Cluster Submission"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Logs directory: $LOGS_DIR"
echo ""

# Determine job array range
if [ "$COMPONENT_ONLY" = true ]; then
    ARRAY_RANGE="1-35"
    DESCRIPTION="component ablations only"
elif [ "$SEQUENCE_ONLY" = true ]; then
    ARRAY_RANGE="36-60"
    DESCRIPTION="sequence length ablations only"
else
    ARRAY_RANGE="1-60"
    DESCRIPTION="all ablation experiments"
fi

echo "📊 Submitting $DESCRIPTION"
echo "   Array range: $ARRAY_RANGE"
echo "   Total jobs: $(echo $ARRAY_RANGE | sed 's/-/ /' | awk '{print $2 - $1 + 1}')"

# Submit array job
ARRAY_SCRIPT="$SCRIPT_DIR/submit_ablation_array.pbs"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - Commands that would be executed:"
    echo "qsub -J $ARRAY_RANGE $ARRAY_SCRIPT"
else
    echo ""
    echo "🚀 Submitting array job..."
    
    # Modify the array range in the PBS script temporarily
    TEMP_SCRIPT=$(mktemp)
    sed "s/#PBS -J 1-60/#PBS -J $ARRAY_RANGE/" "$ARRAY_SCRIPT" > "$TEMP_SCRIPT"
    
    ARRAY_JOB_ID=$(qsub "$TEMP_SCRIPT" | cut -d'.' -f1)
    rm "$TEMP_SCRIPT"
    
    echo "✅ Array job submitted: $ARRAY_JOB_ID"
    echo "   Range: $ARRAY_RANGE"
    
    # Submit analysis job with dependency
    if [ "$COMPONENT_ONLY" = false ] && [ "$SEQUENCE_ONLY" = false ]; then
        echo ""
        echo "📊 Submitting analysis job..."
        
        ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_analysis.pbs"
        ANALYSIS_JOB_ID=$(qsub -W depend=afteranyarray:$ARRAY_JOB_ID "$ANALYSIS_SCRIPT" | cut -d'.' -f1)
        
        echo "✅ Analysis job submitted: $ANALYSIS_JOB_ID"
        echo "   Dependency: afteranyarray:$ARRAY_JOB_ID"
    fi
fi

echo ""
echo "📋 Job Summary:"
echo "==============="

if [ "$DRY_RUN" = false ]; then
    echo "Array Job ID: $ARRAY_JOB_ID"
    if [ "$COMPONENT_ONLY" = false ] && [ "$SEQUENCE_ONLY" = false ]; then
        echo "Analysis Job ID: $ANALYSIS_JOB_ID"
    fi
    echo ""
    echo "Monitor jobs with:"
    echo "  qstat -u \$USER"
    echo "  qstat -t $ARRAY_JOB_ID  # Array job details"
    echo ""
    echo "Check logs in:"
    echo "  $LOGS_DIR/"
    echo ""
    echo "Results will be saved to:"
    echo "  $PROJECT_ROOT/models/ablation/results/"
    echo "  $PROJECT_ROOT/models/ablation/plots/"
else
    echo "DRY RUN - No jobs submitted"
fi

echo ""
echo "🎯 Ablation Study Configuration:"
echo "   Target: M5-class, 72h prediction window"
echo "   Component ablations: 7 variants × 5 seeds = 35 experiments"
echo "   Sequence ablations: 5 variants × 5 seeds = 25 experiments"
echo "   Total experiments: 60"
echo "   Expected runtime: ~24 hours per job"
echo ""
echo "📊 Expected outputs:"
echo "   - Statistical significance tests (paired bootstrap)"
echo "   - Summary tables (CSV format)"
echo "   - Visualization plots (PNG format)"
echo "   - Raw results (JSON format)"

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "✅ Submission complete! Jobs are now queued."
    echo "   Use 'qstat -u \$USER' to monitor progress"
fi 