#!/bin/bash

# Submit script for Imperial RCS HPO jobs
# Usage: ./submit_jobs.sh [setup|single|array|cpu|all]

set -e

# Check if we're in the project root directory
if [[ ! -f "models/hpo/run_hpo.py" ]]; then
    echo "❌ Error: Must run from project root directory (~/masters-project/)"
    echo "Current directory: $(pwd)"
    echo "Expected to find: models/hpo/run_hpo.py"
    echo ""
    echo "Fix: cd ~/masters-project && ./models/hpo/cluster/submit_jobs.sh $@"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on the cluster
if [[ ! $(hostname) == *"login"* ]]; then
    print_warning "You should run this script from a login node"
fi

# Function to submit setup job
submit_setup() {
    print_status "Submitting environment setup job..."
    job_id=$(qsub models/hpo/cluster/setup_environment.pbs)
    print_status "Setup job submitted: $job_id"
    echo $job_id
}

# Function to submit single HPO job
submit_single() {
    local flare_class=${1:-"M"}
    local time_window=${2:-24}
    
    print_status "Submitting single HPO job for ${flare_class} class, ${time_window}h window..."
    
    # Set environment variables for the job
    export FLARE_CLASS=$flare_class
    export TIME_WINDOW=$time_window
    
    job_id=$(qsub -v FLARE_CLASS=$flare_class,TIME_WINDOW=$time_window models/hpo/cluster/run_hpo_single.pbs)
    print_status "Single HPO job submitted: $job_id"
    echo $job_id
}

# Function to submit array job
submit_array() {
    local setup_job_id=$1
    
    print_status "Submitting HPO array job for all 9 target configurations..."
    
    if [ -n "$setup_job_id" ]; then
        job_id=$(qsub -W depend=afterok:$setup_job_id models/hpo/cluster/run_hpo_array.pbs)
        print_status "Array job submitted with dependency: $job_id"
    else
        job_id=$(qsub models/hpo/cluster/run_hpo_array.pbs)
        print_status "Array job submitted: $job_id"
    fi
    
    echo $job_id
}

# Function to submit CPU job
submit_cpu() {
    local flare_class=${1:-"M"}
    local time_window=${2:-24}
    
    print_status "Submitting CPU-only HPO job for ${flare_class} class, ${time_window}h window..."
    
    job_id=$(qsub -v FLARE_CLASS=$flare_class,TIME_WINDOW=$time_window models/hpo/cluster/run_hpo_cpu.pbs)
    print_status "CPU HPO job submitted: $job_id"
    echo $job_id
}

# Main execution
case "${1:-help}" in
    "setup")
        submit_setup
        ;;
    "single")
        submit_single ${2} ${3}
        ;;
    "array")
        if [ -f ".last_setup_job" ]; then
            setup_job_id=$(cat .last_setup_job)
            print_status "Using setup job dependency: $setup_job_id"
            submit_array $setup_job_id
        else
            print_warning "No setup job found. Submitting array job without dependency."
            submit_array
        fi
        ;;
    "cpu")
        submit_cpu ${2} ${3}
        ;;
    "all")
        print_status "Setting up complete HPO workflow..."
        
        # Submit setup job
        setup_job_id=$(submit_setup)
        echo $setup_job_id > .last_setup_job
        
        # Submit array job with dependency
        array_job_id=$(submit_array $setup_job_id)
        
        print_status "Workflow submitted:"
        print_status "  Setup job: $setup_job_id"
        print_status "  Array job: $array_job_id (depends on setup)"
        ;;
    "status")
        print_status "Checking job status..."
        qstat -u $USER
        ;;
    "help"|*)
        echo "Usage: $0 [command] [args...]"
        echo ""
        echo "Commands:"
        echo "  setup                    - Submit environment setup job"
        echo "  single [class] [window]  - Submit single HPO job (default: M 24)"
        echo "  array                    - Submit array job for all 9 targets"
        echo "  cpu [class] [window]     - Submit CPU-only job (default: M 24)"
        echo "  all                      - Submit complete workflow (setup + array)"
        echo "  status                   - Check job status"
        echo "  help                     - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 setup"
        echo "  $0 single M 24"
        echo "  $0 array"
        echo "  $0 all"
        ;;
esac 