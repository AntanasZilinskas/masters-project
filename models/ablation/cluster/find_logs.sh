#!/bin/bash

# Script to find and analyze cluster job logs
# Usage: ./find_logs.sh [optional_job_pattern]

echo "🔍 Searching for cluster job logs..."
echo "=================================="

# Function to check and display logs in a directory
check_directory() {
    local dir="$1"
    local desc="$2"
    
    if [ -d "$dir" ]; then
        echo
        echo "📁 Checking $desc: $dir"
        echo "----------------------------------------"
        
        # Look for .o and .e files
        local output_files=$(find "$dir" -maxdepth 1 -name "*.o*" 2>/dev/null | head -10)
        local error_files=$(find "$dir" -maxdepth 1 -name "*.e*" 2>/dev/null | head -10)
        
        if [ -n "$output_files" ] || [ -n "$error_files" ]; then
            echo "✅ Found log files:"
            
            if [ -n "$output_files" ]; then
                echo "📄 Output files (.o*):"
                echo "$output_files" | while read file; do
                    if [ -f "$file" ]; then
                        local size=$(ls -lh "$file" | awk '{print $5}')
                        local date=$(ls -l "$file" | awk '{print $6, $7, $8}')
                        echo "   $file ($size, $date)"
                    fi
                done
            fi
            
            if [ -n "$error_files" ]; then
                echo "🚨 Error files (.e*):"
                echo "$error_files" | while read file; do
                    if [ -f "$file" ]; then
                        local size=$(ls -lh "$file" | awk '{print $5}')
                        local date=$(ls -l "$file" | awk '{print $6, $7, $8}')
                        echo "   $file ($size, $date)"
                    fi
                done
            fi
        else
            echo "❌ No log files found"
        fi
    else
        echo "❌ Directory not found: $dir"
    fi
}

# Check common locations
check_directory "$(pwd)" "Current directory"
check_directory "$(dirname $(pwd))" "Parent directory"
check_directory "$HOME" "Home directory"
check_directory "/tmp" "Temp directory"

# Search more broadly
echo
echo "🔍 Broad search for recent log files..."
echo "======================================"

# Find log files modified in the last 7 days
recent_logs=$(find ~ -name "*.o*" -o -name "*.e*" -newer $(date -d "7 days ago" +%Y%m%d) 2>/dev/null | head -20)

if [ -n "$recent_logs" ]; then
    echo "✅ Recent log files (last 7 days):"
    echo "$recent_logs" | while read file; do
        if [ -f "$file" ]; then
            local size=$(ls -lh "$file" | awk '{print $5}')
            local date=$(ls -l "$file" | awk '{print $6, $7, $8}')
            echo "   $file ($size, $date)"
        fi
    done
else
    echo "❌ No recent log files found"
fi

# Check job status
echo
echo "📊 Current job status..."
echo "======================="

if command -v qstat >/dev/null 2>&1; then
    echo "✅ Checking current jobs:"
    qstat -u $USER 2>/dev/null || echo "❌ No jobs found or qstat not available"
    
    echo
    echo "📜 Recent job history:"
    qstat -x -u $USER 2>/dev/null | tail -10 || echo "❌ Job history not available"
else
    echo "❌ qstat command not available (not on cluster?)"
fi

# Function to analyze a log file
analyze_log() {
    local file="$1"
    echo
    echo "🔬 Analyzing: $file"
    echo "===================="
    
    if [ ! -f "$file" ]; then
        echo "❌ File not found"
        return
    fi
    
    local size=$(ls -lh "$file" | awk '{print $5}')
    echo "📏 Size: $size"
    
    # Check for errors
    local errors=$(grep -i "error\|failed\|exception\|traceback" "$file" 2>/dev/null | head -5)
    if [ -n "$errors" ]; then
        echo "🚨 Errors found:"
        echo "$errors"
    else
        echo "✅ No obvious errors found"
    fi
    
    # Check for completion
    local completion=$(grep -i "complete\|finished\|done\|success" "$file" 2>/dev/null | tail -3)
    if [ -n "$completion" ]; then
        echo "✅ Completion indicators:"
        echo "$completion"
    fi
    
    # Show last few lines
    echo "📝 Last 10 lines:"
    tail -10 "$file" 2>/dev/null || echo "❌ Cannot read file"
}

# If specific pattern provided, analyze matching files
if [ $# -gt 0 ]; then
    pattern="$1"
    echo
    echo "🎯 Analyzing files matching pattern: $pattern"
    echo "============================================="
    
    matching_files=$(find . ~ -name "*$pattern*" \( -name "*.o*" -o -name "*.e*" \) 2>/dev/null | head -5)
    
    if [ -n "$matching_files" ]; then
        echo "$matching_files" | while read file; do
            analyze_log "$file"
        done
    else
        echo "❌ No files found matching pattern: $pattern"
    fi
fi

echo
echo "💡 Tips:"
echo "========"
echo "1. Run with pattern: ./find_logs.sh component_ablation"
echo "2. To analyze specific file: ./find_logs.sh && cat [filename]"
echo "3. To monitor real-time: tail -f [filename]"
echo "4. To search for errors: grep -i error [filename]"
echo "5. For help: see HOW_TO_INSPECT_CLUSTER_LOGS.md" 