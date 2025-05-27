#!/bin/bash

"""
EVEREST Cluster Resource Testing Script

This script tests different resource configurations to identify
what works on your specific cluster system.
"""

echo "🔍 EVEREST Cluster Resource Testing"
echo "=================================="
echo ""

# Check what scheduler is available
if command -v sbatch &> /dev/null; then
    echo "✅ SLURM detected (sbatch available)"
    SCHEDULER="SLURM"
elif command -v qsub &> /dev/null; then
    echo "✅ PBS detected (qsub available)"
    SCHEDULER="PBS"
else
    echo "❌ No job scheduler detected"
    echo "   Neither sbatch (SLURM) nor qsub (PBS) found"
    exit 1
fi

echo ""
echo "📊 Testing resource configurations..."

# Create a minimal test script
TEST_SCRIPT=$(mktemp)
cat > "$TEST_SCRIPT" << 'EOF'
#!/bin/bash
echo "Test job started at $(date)"
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi -L 2>/dev/null || echo "No NVIDIA GPUs found"
echo "Test job completed at $(date)"
EOF

chmod +x "$TEST_SCRIPT"

if [ "$SCHEDULER" = "SLURM" ]; then
    echo ""
    echo "🧪 Testing SLURM configurations..."
    
    # Test basic SLURM configuration
    echo "1. Testing basic SLURM config..."
    SLURM_TEST=$(mktemp)
    cat > "$SLURM_TEST" << EOF
#!/bin/bash
#SBATCH --job-name=everest_test
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=test_slurm_%j.log

$TEST_SCRIPT
EOF
    
    if JOB_ID=$(sbatch "$SLURM_TEST" 2>/dev/null | awk '{print $4}'); then
        echo "   ✅ Basic SLURM config works: Job $JOB_ID"
        echo "   Resource spec: --nodes=1 --cpus-per-task=2 --mem=8G --gres=gpu:1"
    else
        echo "   ❌ Basic SLURM config failed"
    fi
    
    rm "$SLURM_TEST"

elif [ "$SCHEDULER" = "PBS" ]; then
    echo ""
    echo "🧪 Testing PBS configurations..."
    
    # Test configurations in order of preference
    PBS_CONFIGS=(
        "select=1:ncpus=2:mem=8gb:ngpus=1"
        "select=1:ncpus=2:mem=8gb:ngpus=1:gpu_type=L40S"
        "select=1:ncpus=2:mem=8gb:ngpus=1:gpu_type=V100"
        "select=1:ncpus=2:mem=8gb:ngpus=1:gpu_type=A100"
        "nodes=1:ppn=2:gpus=1"
        "nodes=1:ppn=2"
        "select=1:ncpus=2:mem=8gb"
    )
    
    for i in "${!PBS_CONFIGS[@]}"; do
        config="${PBS_CONFIGS[$i]}"
        echo "$((i+1)). Testing PBS config: $config"
        
        PBS_TEST=$(mktemp)
        cat > "$PBS_TEST" << EOF
#!/bin/bash
#PBS -N everest_test_$((i+1))
#PBS -l walltime=00:05:00
#PBS -l $config
#PBS -j oe
#PBS -o test_pbs_$((i+1)).log

$TEST_SCRIPT
EOF
        
        if JOB_ID=$(qsub "$PBS_TEST" 2>/dev/null | cut -d'.' -f1); then
            echo "   ✅ Config $((i+1)) works: Job $JOB_ID"
            echo "   Resource spec: $config"
            WORKING_CONFIG="$config"
            rm "$PBS_TEST"
            break
        else
            echo "   ❌ Config $((i+1)) failed"
        fi
        
        rm "$PBS_TEST"
    done
    
    if [ -z "$WORKING_CONFIG" ]; then
        echo ""
        echo "❌ All PBS configurations failed!"
        echo ""
        echo "🔧 Manual testing suggestions:"
        echo "1. Check available resources:"
        echo "   qstat -Q"
        echo "   pbsnodes -a"
        echo ""
        echo "2. Check cluster documentation for correct syntax"
        echo ""
        echo "3. Try submitting a simple job manually:"
        echo "   echo 'echo hello' | qsub -l walltime=00:01:00"
        echo ""
        echo "4. Contact your cluster administrator for help"
    fi
fi

# Cleanup
rm "$TEST_SCRIPT"

echo ""
echo "📋 Summary:"
echo "==========="
echo "Scheduler: $SCHEDULER"

if [ "$SCHEDULER" = "PBS" ] && [ -n "$WORKING_CONFIG" ]; then
    echo "Working PBS config: $WORKING_CONFIG"
    echo ""
    echo "🔧 To use this configuration in the ablation study:"
    echo "1. Edit models/ablation/cluster/submit_ablation_array.pbs"
    echo "2. Change the resource line to:"
    echo "   #PBS -l $WORKING_CONFIG"
    echo "3. Run: ./submit_jobs.sh"
fi

echo ""
echo "🎯 Next steps:"
echo "1. Wait for test job to complete"
echo "2. Check test job logs for any issues"
echo "3. If successful, proceed with ablation study submission"

if [ "$SCHEDULER" = "PBS" ]; then
    echo "4. Monitor with: qstat -u $USER"
elif [ "$SCHEDULER" = "SLURM" ]; then
    echo "4. Monitor with: squeue -u $USER"
fi 