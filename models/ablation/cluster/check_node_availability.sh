#!/bin/bash

# Check Node Availability and Queue Status for ICL Cluster
# This script helps determine the best submission strategy

echo "🔍 ICL Cluster Resource Availability Check"
echo "=========================================="
echo ""

# Check current queue status
echo "📊 Current Queue Status:"
echo "------------------------"
qstat -Q v1_gpu72 2>/dev/null || echo "Queue info not available"
echo ""

# Check your current jobs
echo "👤 Your Current Jobs:"
echo "--------------------"
USER_JOBS=$(qstat -u $USER 2>/dev/null)
if [ -n "$USER_JOBS" ]; then
    echo "$USER_JOBS"
    JOB_COUNT=$(echo "$USER_JOBS" | grep -c "^[0-9]" || echo "0")
    echo ""
    echo "Total jobs: $JOB_COUNT"
else
    echo "No jobs currently running"
    JOB_COUNT=0
fi
echo ""

# Check node availability
echo "🖥️  Node Availability:"
echo "---------------------"

# Try to get node information
NODES_INFO=$(pbsnodes -a 2>/dev/null | grep -E "(Node Id|state|resources_available|resources_assigned)" || echo "Node info not available")

if [ "$NODES_INFO" != "Node info not available" ]; then
    echo "$NODES_INFO" | head -20
    echo "..."
    echo ""
    
    # Count available nodes
    AVAILABLE_NODES=$(pbsnodes -a 2>/dev/null | grep -c "state = free" || echo "0")
    BUSY_NODES=$(pbsnodes -a 2>/dev/null | grep -c "state = job-busy" || echo "0")
    
    echo "Available nodes: $AVAILABLE_NODES"
    echo "Busy nodes: $BUSY_NODES"
else
    echo "Node information not accessible"
fi
echo ""

# Check GPU availability specifically
echo "🎮 GPU Availability:"
echo "-------------------"
GPU_NODES=$(pbsnodes -a 2>/dev/null | grep -A 10 -B 2 "ngpus" | head -20 || echo "GPU info not available")
echo "$GPU_NODES"
echo ""

# Recommendations based on current status
echo "💡 Recommendations:"
echo "------------------"

if [ $JOB_COUNT -eq 0 ]; then
    echo "✅ No current jobs - good time to submit"
    
    if [ $AVAILABLE_NODES -gt 0 ] 2>/dev/null; then
        echo "✅ Available nodes detected"
        echo ""
        echo "🎯 RECOMMENDED: Try whole node approach first"
        echo "   qsub models/ablation/cluster/submit_whole_node.pbs"
        echo ""
        echo "🔄 FALLBACK: If whole node rejected, use sequential"
        echo "   qsub models/ablation/cluster/submit_sequential_batch.pbs"
    else
        echo "⚠️  Limited node availability"
        echo ""
        echo "🎯 RECOMMENDED: Use sequential approach"
        echo "   qsub models/ablation/cluster/submit_sequential_batch.pbs"
    fi
    
elif [ $JOB_COUNT -lt 5 ]; then
    echo "⚠️  You have $JOB_COUNT jobs running"
    echo ""
    echo "🎯 RECOMMENDED: Wait for jobs to complete or use small batch"
    echo "   qsub models/ablation/cluster/submit_ablation_small.pbs"
    
else
    echo "❌ You have $JOB_COUNT jobs running - consider waiting"
    echo ""
    echo "🎯 RECOMMENDED: Wait for current jobs to complete"
fi

echo ""

# Test submissions (dry run)
echo "🧪 Test Submission Commands:"
echo "----------------------------"
echo ""
echo "# Test whole node availability:"
echo "qsub -W depend=afternotok models/ablation/cluster/submit_whole_node.pbs"
echo ""
echo "# Test sequential submission:"
echo "qsub models/ablation/cluster/submit_sequential_batch.pbs"
echo ""
echo "# Check what would be submitted:"
echo "qsub -n models/ablation/cluster/submit_whole_node.pbs"

echo ""
echo "📋 Quick Commands:"
echo "-----------------"
echo "# Monitor queue: watch -n 30 'qstat -u $USER'"
echo "# Check nodes:   pbsnodes -a | grep -E '(Node Id|state|ngpus)'"
echo "# Queue limits:  qstat -Q" 