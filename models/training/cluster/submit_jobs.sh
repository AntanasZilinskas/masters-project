#!/bin/bash

# EVEREST Production Training - Cluster Job Submission Script
#
# This script submits the production training array job and optional analysis job.
#
# Usage:
#   ./submit_jobs.sh [--dry-run] [--targets C-24 M-48 M5-72] [--no-analysis]

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/models/training/logs"

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
            # Collect all arguments until we hit another option or end of args
            while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
                if [ -z "$TARGETS_FILTER" ]; then
                    TARGETS_FILTER="$1"
                else
                    TARGETS_FILTER="$TARGETS_FILTER $1"
                fi
                shift
            done
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
echo "   Resource: 1 GPU, 4 cores, 24GB RAM per job"
echo "   Time limit: 24 hours per job"

# Submit array job
ARRAY_SCRIPT="$SCRIPT_DIR/submit_production_array.pbs"

if [ ! -f "$ARRAY_SCRIPT" ]; then
    echo "❌ Error: Array script not found: $ARRAY_SCRIPT"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "🔍 DRY RUN - Commands that would be executed:"
    echo "qsub $ARRAY_SCRIPT"
    
    if [ "$NO_ANALYSIS" = false ]; then
        echo "# After array job completes:"
        echo "qsub -W depend=afterokarray:<ARRAY_JOB_ID> $SCRIPT_DIR/submit_analysis.pbs"
    fi
else
    echo ""
    echo "🚀 Submitting array job..."
    
    # Submit the array job directly (range is already set in the PBS script)
    if ARRAY_JOB_ID=$(qsub "$ARRAY_SCRIPT" 2>&1); then
        # Extract job ID from PBS output (format may vary by system)
        ARRAY_JOB_ID=$(echo "$ARRAY_JOB_ID" | grep -o '[0-9]\+' | head -1)
        echo "✅ Array job submitted: $ARRAY_JOB_ID"
        echo "   Range: $ARRAY_RANGE"
    else
        echo "❌ Failed to submit array job"
        echo "Error: $ARRAY_JOB_ID"
        exit 1
    fi
    
    # Submit analysis job with dependency
    if [ "$NO_ANALYSIS" = false ]; then
        echo ""
        echo "📊 Submitting analysis job..."
        
        ANALYSIS_SCRIPT="$SCRIPT_DIR/submit_analysis.pbs"
        
        if [ ! -f "$ANALYSIS_SCRIPT" ]; then
            echo "⚠️  Analysis script not found: $ANALYSIS_SCRIPT"
            echo "   Skipping analysis job submission"
        else
            # Try different dependency syntaxes for different PBS systems
            if ANALYSIS_JOB_ID=$(qsub -W depend=afterokarray:$ARRAY_JOB_ID "$ANALYSIS_SCRIPT" 2>/dev/null); then
                ANALYSIS_JOB_ID=$(echo "$ANALYSIS_JOB_ID" | grep -o '[0-9]\+' | head -1)
                echo "✅ Analysis job submitted: $ANALYSIS_JOB_ID"
                echo "   Dependency: afterokarray:$ARRAY_JOB_ID"
            elif ANALYSIS_JOB_ID=$(qsub -W depend=afterany:$ARRAY_JOB_ID "$ANALYSIS_SCRIPT" 2>/dev/null); then
                ANALYSIS_JOB_ID=$(echo "$ANALYSIS_JOB_ID" | grep -o '[0-9]\+' | head -1)
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
echo "   qstat -u \$USER"
if [ "$DRY_RUN" = false ] && [ -n "$ARRAY_JOB_ID" ]; then
    echo "   qstat -t $ARRAY_JOB_ID  # Array job details"
fi
echo "   watch -n 30 'qstat -u \$USER'  # Auto-refresh every 30s"
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

echo ""
echo "🚀 Next steps:"
echo "   1. Monitor job progress with: qstat -u \$USER"
echo "   2. Check individual job logs in: models/training/logs/"
echo "   3. View results after completion in: models/training/results/" 