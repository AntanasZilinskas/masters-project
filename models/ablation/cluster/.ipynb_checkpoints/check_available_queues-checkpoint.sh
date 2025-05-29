#!/bin/bash

# Check Available Queues on Imperial RCS
# Run this script on the cluster to see what queues are actually available

echo "🔍 Checking Available Queues on Imperial RCS"
echo "============================================="
echo ""

echo "📋 All Available Queues:"
echo "------------------------"
qstat -Q 2>/dev/null || echo "qstat -Q failed - trying alternative methods"
echo ""

echo "📋 Queue Details:"
echo "----------------"
qstat -Qf 2>/dev/null | head -50 || echo "qstat -Qf failed"
echo ""

echo "🎮 GPU-Related Queues:"
echo "---------------------"
qstat -Q 2>/dev/null | grep -i gpu || echo "No GPU queues found with 'gpu' in name"
echo ""

echo "📊 Queue Information:"
echo "--------------------"
# Try to get more detailed queue info
pbsnodes -a 2>/dev/null | grep -E "(Queue|queue)" | head -10 || echo "pbsnodes queue info not available"
echo ""

echo "🔧 Alternative Queue Discovery:"
echo "------------------------------"
# Check if there are any job examples or documentation
ls /opt/pbs/share/doc/ 2>/dev/null | grep -i queue || echo "No PBS documentation found"
echo ""

# Check what queues are mentioned in any existing jobs
echo "📝 Queues from Recent Jobs:"
echo "--------------------------"
qstat -f $(qstat | tail -n +3 | head -5 | awk '{print $1}' | cut -d'.' -f1) 2>/dev/null | grep -i queue || echo "No recent jobs to check"
echo ""

echo "💡 Recommendations:"
echo "------------------"
echo "1. Try common queue names:"
echo "   - gpu"
echo "   - gpu72" 
echo "   - v1_gpu72"
echo "   - batch"
echo "   - default"
echo ""
echo "2. Contact Imperial RCS support for current queue names"
echo "3. Check cluster documentation or examples"
echo ""

echo "🧪 Test Queue Submissions:"
echo "-------------------------"
echo "# Test if basic gpu queue exists:"
echo "qsub -q gpu -l select=1:ncpus=1:mem=1gb:ngpus=1 -l walltime=00:01:00 -- /bin/echo 'test'"
echo ""
echo "# Test if v1_gpu72 still works:"
echo "qsub -q v1_gpu72 -l select=1:ncpus=1:mem=1gb:ngpus=1 -l walltime=00:01:00 -- /bin/echo 'test'"
echo ""
echo "# Test without specifying queue (use default):"
echo "qsub -l select=1:ncpus=1:mem=1gb:ngpus=1 -l walltime=00:01:00 -- /bin/echo 'test'" 