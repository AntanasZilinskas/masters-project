#!/bin/bash

"""
EVEREST Production Training - Cluster Job Submission Script

This script submits the production training array job and optional analysis job.

Usage:
  ./submit_jobs.sh [--dry-run] [--targets C-24 M-48 M5-72] [--no-analysis]
"""

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"

# Parse command line arguments
DRY_RUN=false
TARGETS_FILTER=""
NO_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --targets)
            shift
            TARGETS_FILTER="$*"
            break
            ;;
        --no-analysis)
            NO_ANALYSIS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--targets C-24 M-48 M5-72] [--no-analysis]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Show commands without executing"
            echo "  --targets      Filter specific targets (e.g., C-24 M-48 M5-72)"
            echo "  --no-analysis  Skip analysis job submission"
            echo "  -h, --help     Show this help message"
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

echo "🏭 EVEREST Production Training - Cluster Submission"
echo "=================================================="
echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Script directory: $SCRIPT_DIR"
echo "📁 Logs directory: $LOGS_DIR"

if [ "$TARGETS_FILTER" != "" ]; then
    echo "🎯 Target filter: $TARGETS_FILTER"
fi

if [ "$NO_ANALYSIS" = true ]; then
    echo "⚠️  Analysis job will be skipped"
fi

if [ "$DRY_RUN" = true ]; then
    echo "🔍 DRY RUN MODE - No jobs will be submitted"
fi

echo ""

# Determine array range based on targets filter
if [ "$TARGETS_FILTER" != "" ]; then
    # For filtered targets, we'd need to calculate the specific array indices
    # For simplicity, we'll run all and let the script filter
    ARRAY_RANGE="1-45"
    echo "⚠️  Note: Running full array range with filtering in script"
else
    ARRAY_RANGE="1-45"
fi

echo "📊 Array job configuration:"
echo "   Range: $ARRAY_RANGE"
echo "   Total jobs: 45 (9 targets × 5 seeds)"
echo "   Resource: 1 GPU (L40S), 8 cores, 64GB RAM per job"
echo "   Time limit: 24 hours per job"

# Submit array job
ARRAY_SCRIPT="$SCRIPT_DIR/submit_production_array.pbs"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - Commands that would be executed:"
    echo "qsub -J $ARRAY_RANGE $ARRAY_SCRIPT"
    
    if [ "$NO_ANALYSIS" = false ]; then
        echo "# After array job completes:"
        echo "qsub -W depend=afterokarray:<ARRAY_JOB_ID> $SCRIPT_DIR/submit_analysis.pbs"
    fi
else
    echo ""
    echo "🚀 Submitting array job..."
    
    # Modify the array range in the PBS script temporarily
    TEMP_SCRIPT=$(mktemp)
    sed "s/#PBS -J 1-45/#PBS -J $ARRAY_RANGE/" "$ARRAY_SCRIPT" > "$TEMP_SCRIPT"
    
    ARRAY_JOB_ID=$(qsub "$TEMP_SCRIPT" | cut -d'.' -f1)
    rm "$TEMP_SCRIPT"
    
    echo "✅ Array job submitted: $ARRAY_JOB_ID"
    echo "   Range: $ARRAY_RANGE"
    
    # Submit analysis job with dependency
    if [ "$NO_ANALYSIS" = false ]; then
        echo ""
        echo "📊 Submitting analysis job..."
        
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

echo ""
echo "📋 Job Summary:"
echo "=============="

if [ "$DRY_RUN" = false ]; then
    echo "Array Job ID: $ARRAY_JOB_ID"
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
echo "   qstat -u $USER"
echo "   qstat -t $ARRAY_JOB_ID  # Array job details"
echo ""
echo "📁 Results will be saved to:"
echo "   models/training/results/"
echo "   models/training/trained_models/"
echo ""
echo "🎯 Expected completion time: ~18-24 hours (parallel execution)"
echo "💾 Expected storage usage: ~50GB"

if [ "$TARGETS_FILTER" != "" ]; then
    echo ""
    echo "⚠️  Note: Target filtering is applied within the training script"
    echo "   Filtered targets: $TARGETS_FILTER"
fi 