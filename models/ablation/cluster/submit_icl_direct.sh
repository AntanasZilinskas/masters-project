#!/bin/bash

# Direct submission script for ICL cluster
# This bypasses the complex fallback logic and uses the known working configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "🧪 EVEREST Ablation Study - ICL Direct Submission"
echo "================================================"
echo ""
echo "📁 Project root: $PROJECT_ROOT"
echo "📁 Script directory: $SCRIPT_DIR"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Change to project root for submission
cd "$PROJECT_ROOT"

echo ""
echo "📊 Submitting ablation study array job..."
echo "   Queue: v1_gpu72"
echo "   Jobs: 1-60"
echo "   Resources: 1 GPU, 4 cores, 24GB RAM"
echo "   Time limit: 24 hours"

# Submit the ICL-specific PBS script
PBS_SCRIPT="$SCRIPT_DIR/submit_ablation_array_icl.pbs"

if [ ! -f "$PBS_SCRIPT" ]; then
    echo "❌ PBS script not found: $PBS_SCRIPT"
    exit 1
fi

echo ""
echo "🚀 Submitting job..."

# Submit the job and capture output
SUBMIT_OUTPUT=$(qsub "$PBS_SCRIPT" 2>&1)
SUBMIT_EXIT_CODE=$?

if [ $SUBMIT_EXIT_CODE -eq 0 ]; then
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | cut -d'.' -f1)
    echo "✅ Array job submitted successfully!"
    echo "   Job ID: $JOB_ID"
    echo "   Full output: $SUBMIT_OUTPUT"
    
    echo ""
    echo "📊 Monitoring commands:"
    echo "   qstat -u $USER"
    echo "   qstat -t $JOB_ID"
    
    echo ""
    echo "📁 Logs will be saved to:"
    echo "   logs/ablation_*.log"
    
    echo ""
    echo "📈 Results will be saved to:"
    echo "   models/ablation/results/"
    echo "   models/ablation/trained_models/"
    
    echo ""
    echo "🎯 Expected completion: ~24 hours"
    echo "💾 Expected storage: ~30GB"
    
else
    echo "❌ Job submission failed!"
    echo "   Exit code: $SUBMIT_EXIT_CODE"
    echo "   Error output: $SUBMIT_OUTPUT"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "1. Check if you're in the correct directory"
    echo "2. Verify the queue name: qstat -Q"
    echo "3. Check your account limits: qstat -u $USER"
    echo "4. Try manual submission:"
    echo "   qsub $PBS_SCRIPT"
    exit 1
fi 