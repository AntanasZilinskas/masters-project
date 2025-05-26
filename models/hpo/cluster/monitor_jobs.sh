#!/bin/bash

# Job monitoring script for Imperial RCS HPO jobs
# Usage: ./monitor_jobs.sh [watch|summary|logs|cleanup]

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

# Function to show job status
show_status() {
    print_header "Current Job Status"
    qstat -u $USER
    echo ""
    
    # Count jobs by status
    local running=$(qstat -u $USER 2>/dev/null | grep -c " R " || echo "0")
    local queued=$(qstat -u $USER 2>/dev/null | grep -c " Q " || echo "0")
    local total=$(qstat -u $USER 2>/dev/null | tail -n +3 | wc -l || echo "0")
    
    echo "Summary: $total total jobs ($running running, $queued queued)"
}

# Function to show detailed summary
show_summary() {
    print_header "HPO Job Summary"
    
    # Check for results directories
    if [ -d "results" ]; then
        echo "Results directories:"
        ls -la results/ | grep "^d" | awk '{print "  " $9}' | grep "hpo_" || echo "  No HPO results found"
        echo ""
    fi
    
    # Check for log files
    echo "Recent job logs:"
    ls -lt *.out *.err 2>/dev/null | head -10 || echo "  No log files found"
    echo ""
    
    # Check virtual environment
    if [ -d "venv_hpo" ]; then
        print_status "Virtual environment exists at: $(pwd)/venv_hpo"
    else
        print_warning "Virtual environment not found. Run setup first."
    fi
}

# Function to watch jobs in real-time
watch_jobs() {
    print_header "Watching Jobs (Press Ctrl+C to exit)"
    while true; do
        clear
        show_status
        echo ""
        echo "Last updated: $(date)"
        echo "Press Ctrl+C to exit..."
        sleep 30
    done
}

# Function to show recent logs
show_logs() {
    local job_pattern=${1:-"hpo"}
    
    print_header "Recent Log Output"
    
    # Find most recent output files
    local latest_out=$(ls -t *${job_pattern}*.out 2>/dev/null | head -1)
    local latest_err=$(ls -t *${job_pattern}*.err 2>/dev/null | head -1)
    
    if [ -n "$latest_out" ]; then
        echo -e "${GREEN}Latest stdout ($latest_out):${NC}"
        echo "----------------------------------------"
        tail -20 "$latest_out"
        echo ""
    fi
    
    if [ -n "$latest_err" ]; then
        echo -e "${RED}Latest stderr ($latest_err):${NC}"
        echo "----------------------------------------"
        tail -20 "$latest_err"
        echo ""
    fi
    
    if [ -z "$latest_out" ] && [ -z "$latest_err" ]; then
        print_warning "No log files found matching pattern: *${job_pattern}*"
    fi
}

# Function to show resource usage for running jobs
show_resources() {
    print_header "Resource Usage"
    
    # Get running job IDs
    local running_jobs=$(qstat -u $USER 2>/dev/null | grep " R " | awk '{print $1}' | cut -d. -f1)
    
    if [ -z "$running_jobs" ]; then
        print_warning "No running jobs found"
        return
    fi
    
    for job_id in $running_jobs; do
        echo "Job $job_id:"
        qstat -f $job_id 2>/dev/null | grep -E "(Job_Name|resources_used|Resource_List)" | sed 's/^/  /'
        echo ""
    done
}

# Function to clean up old files
cleanup() {
    print_header "Cleanup"
    
    # Ask for confirmation
    echo "This will remove:"
    echo "  - Log files older than 7 days (*.out, *.err)"
    echo "  - Empty result directories"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up old log files..."
        find . -name "*.out" -o -name "*.err" -mtime +7 -delete 2>/dev/null || true
        
        print_status "Removing empty result directories..."
        find results/ -type d -empty -delete 2>/dev/null || true
        
        print_status "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  status      - Show current job status (default)"
    echo "  watch       - Watch job status in real-time"
    echo "  summary     - Show detailed summary of jobs and results"
    echo "  logs [pattern] - Show recent log output (default pattern: 'hpo')"
    echo "  resources   - Show resource usage for running jobs"
    echo "  cleanup     - Clean up old log files and empty directories"
    echo "  help        - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 watch"
    echo "  $0 logs setup"
    echo "  $0 logs array"
}

# Main execution
case "${1:-status}" in
    "status")
        show_status
        ;;
    "watch")
        watch_jobs
        ;;
    "summary")
        show_summary
        ;;
    "logs")
        show_logs ${2}
        ;;
    "resources")
        show_resources
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac 