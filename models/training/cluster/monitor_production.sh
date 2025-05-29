#!/bin/bash

# Production Training Job Monitor
# Usage: ./monitor_production.sh [status|logs|summary|watch]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show job status
show_status() {
    print_header "Production Training Job Status"
    
    # Check if any production training jobs exist
    if qstat -u $USER 2>/dev/null | grep -q "production_training"; then
        echo "Current production training jobs:"
        qstat -u $USER | grep -E "(Job id|production_training)"
        echo ""
        
        # Count jobs by status
        local running=$(qstat -u $USER 2>/dev/null | grep "production_training" | grep -c " R " || echo "0")
        local queued=$(qstat -u $USER 2>/dev/null | grep "production_training" | grep -c " Q " || echo "0")
        local total=$(qstat -u $USER 2>/dev/null | grep "production_training" | wc -l || echo "0")
        
        echo "Summary: $total production jobs ($running running, $queued queued)"
    else
        echo "No production training jobs currently in queue"
        
        # Check for completed jobs by looking at log files
        if ls production_training.o* >/dev/null 2>&1; then
            echo "Found completed job log files:"
            ls -lt production_training.o* | head -5
        fi
    fi
}

# Function to show log summary
show_logs() {
    print_header "Production Training Log Summary"
    
    if ! ls production_training.o* >/dev/null 2>&1; then
        print_warning "No production training log files found"
        return
    fi
    
    echo "Recent log files:"
    ls -lt production_training.o* production_training.e* 2>/dev/null | head -10
    echo ""
    
    # Show latest output
    latest_out=$(ls -t production_training.o* 2>/dev/null | head -1)
    if [ -n "$latest_out" ]; then
        echo -e "${GREEN}Latest output ($latest_out):${NC}"
        echo "----------------------------------------"
        tail -20 "$latest_out"
        echo ""
        
        # Check corresponding error file
        error_file="${latest_out//.o/.e}"
        if [ -f "$error_file" ] && [ -s "$error_file" ]; then
            echo -e "${RED}Latest errors ($error_file):${NC}"
            echo "----------------------------------------"
            tail -10 "$error_file"
            echo ""
        fi
    fi
}

# Function to show comprehensive summary
show_summary() {
    print_header "Production Training Summary"
    
    # Job status
    show_status
    echo ""
    
    # Log file analysis
    if ls production_training.o* >/dev/null 2>&1; then
        echo "=== JOB COMPLETION ANALYSIS ==="
        
        local total_logs=$(ls production_training.o* 2>/dev/null | wc -l)
        local successful=$(grep -l "completed successfully" production_training.o* 2>/dev/null | wc -l)
        local failed=$(grep -l "failed" production_training.o* 2>/dev/null | wc -l)
        local running=$((total_logs - successful - failed))
        
        echo "Total job logs found: $total_logs"
        echo "✅ Successful: $successful"
        echo "❌ Failed: $failed"
        echo "⏳ Running/Unknown: $running"
        echo ""
        
        if [ $successful -gt 0 ]; then
            echo "✅ SUCCESSFUL JOBS:"
            grep -l "completed successfully" production_training.o* 2>/dev/null | while read file; do
                array_index=$(echo "$file" | sed 's/.*\.//')
                experiment_name=$(grep "experiment_name" "$file" 2>/dev/null | head -1 || echo "unknown")
                echo "  Job $array_index: $experiment_name"
            done
            echo ""
        fi
        
        if [ $failed -gt 0 ]; then
            echo "❌ FAILED JOBS:"
            grep -l "failed" production_training.o* 2>/dev/null | while read file; do
                array_index=$(echo "$file" | sed 's/.*\.//')
                error_msg=$(grep -A2 -B2 "failed\|Error\|Exception" "$file" 2>/dev/null | head -1 || echo "unknown error")
                echo "  Job $array_index: $error_msg"
            done
            echo ""
        fi
        
        # Check for results
        echo "=== RESULTS CHECK ==="
        if ls ../../EVEREST-v* >/dev/null 2>&1; then
            model_count=$(ls -d ../../EVEREST-v* 2>/dev/null | wc -l)
            echo "✅ Trained models found: $model_count"
            echo "Latest models:"
            ls -lt ../../EVEREST-v* | head -5 | awk '{print "  " $9}'
        else
            echo "⚠️  No trained models found yet"
        fi
        
        if [ -d "../results" ] && ls ../results/*.json >/dev/null 2>&1; then
            result_count=$(ls ../results/*.json 2>/dev/null | wc -l)
            echo "✅ Result files found: $result_count"
        else
            echo "⚠️  No result files found yet"
        fi
    else
        print_warning "No log files found - jobs may not have started yet"
    fi
}

# Function to watch jobs in real-time
watch_jobs() {
    print_header "Watching Production Training Jobs (Press Ctrl+C to exit)"
    
    while true; do
        clear
        show_summary
        echo ""
        echo "Last updated: $(date)"
        echo "Press Ctrl+C to exit..."
        sleep 30
    done
}

# Function to show detailed job info
show_job_details() {
    local job_num=$1
    
    if [ -z "$job_num" ]; then
        echo "Usage: $0 details <job_number>"
        echo "Example: $0 details 1"
        return 1
    fi
    
    print_header "Job $job_num Details"
    
    # Find log files for this job
    local out_file=$(ls production_training.o*.$job_num 2>/dev/null | head -1)
    local err_file=$(ls production_training.e*.$job_num 2>/dev/null | head -1)
    
    if [ -z "$out_file" ]; then
        print_error "No log files found for job $job_num"
        return 1
    fi
    
    echo "Output file: $out_file"
    echo "Error file: $err_file"
    echo ""
    
    # Show job status
    if grep -q "completed successfully" "$out_file"; then
        echo -e "${GREEN}Status: ✅ SUCCESS${NC}"
        
        # Extract key metrics
        echo ""
        echo "Results:"
        grep -E "(experiment_name|TSS|Accuracy|F1|threshold)" "$out_file" | sed 's/^/  /'
        
    elif grep -q "failed" "$out_file"; then
        echo -e "${RED}Status: ❌ FAILED${NC}"
        
        # Show error details
        echo ""
        echo "Error details:"
        grep -A5 -B5 "failed\|Error\|Exception" "$out_file" | sed 's/^/  /'
        
        if [ -f "$err_file" ] && [ -s "$err_file" ]; then
            echo ""
            echo "Error file contents:"
            cat "$err_file" | sed 's/^/  /'
        fi
    else
        echo -e "${YELLOW}Status: ⏳ RUNNING or UNKNOWN${NC}"
    fi
    
    echo ""
    echo "Full output (last 50 lines):"
    echo "----------------------------------------"
    tail -50 "$out_file"
}

# Main script logic
case "${1:-status}" in
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "summary")
        show_summary
        ;;
    "watch")
        watch_jobs
        ;;
    "details")
        show_job_details "$2"
        ;;
    "help"|"-h"|"--help")
        echo "Production Training Monitor"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  status    - Show current job status (default)"
        echo "  logs      - Show recent log output"
        echo "  summary   - Show comprehensive summary"
        echo "  watch     - Watch jobs in real-time"
        echo "  details N - Show detailed info for job N"
        echo "  help      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                    # Show status"
        echo "  $0 summary           # Full summary"
        echo "  $0 details 1         # Details for job 1"
        echo "  $0 watch             # Real-time monitoring"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 