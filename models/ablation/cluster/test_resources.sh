#!/bin/bash
"""
Test PBS Resource Configurations for EVEREST Ablation Study

This script tests different resource configurations to find what works
on your specific cluster.
"""

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔧 Testing PBS Resource Configurations"
echo "======================================"
echo ""

# Test configurations
CONFIGS=(
    "select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S"
    "select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=A100"
    "select=1:ncpus=4:mem=24gb:ngpus=1"
    "select=1:ncpus=2:mem=16gb:ngpus=1"
    "select=1:ncpus=2:mem=8gb"
)

CONFIG_NAMES=(
    "L40S GPU (4 cores, 24GB)"
    "A100 GPU (4 cores, 24GB)"
    "Generic GPU (4 cores, 24GB)"
    "Generic GPU (2 cores, 16GB)"
    "CPU-only (2 cores, 8GB)"
)

# Create a simple test job
TEST_JOB=$(mktemp)
cat > "$TEST_JOB" << 'EOF'
#!/bin/bash
#PBS -N test_resources
#PBS -l walltime=00:01:00
#PBS -j oe

echo "Test job running successfully"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
EOF

echo "Testing resource configurations..."
echo ""

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${CONFIG_NAMES[$i]}"
    
    echo -n "[$((i+1))/${#CONFIGS[@]}] Testing $name... "
    
    # Create test job with this configuration
    TEMP_JOB=$(mktemp)
    sed "s/#PBS -l walltime=00:01:00/#PBS -l walltime=00:01:00\n#PBS -l $config/" "$TEST_JOB" > "$TEMP_JOB"
    
    # Try to submit (dry run)
    if qsub -W depend=afterany:999999 "$TEMP_JOB" >/dev/null 2>&1; then
        echo "✅ SUPPORTED"
    else
        echo "❌ NOT SUPPORTED"
    fi
    
    rm "$TEMP_JOB"
done

rm "$TEST_JOB"

echo ""
echo "💡 Recommendations:"
echo "   - Use the first supported configuration for best performance"
echo "   - GPU configurations are preferred for EVEREST training"
echo "   - CPU-only can be used for testing but will be much slower"
echo ""
echo "📋 To use a specific configuration:"
echo "   1. Edit the PBS scripts in models/ablation/cluster/"
echo "   2. Update the #PBS -l select=... line"
echo "   3. Submit jobs with ./submit_jobs_simple.sh" 