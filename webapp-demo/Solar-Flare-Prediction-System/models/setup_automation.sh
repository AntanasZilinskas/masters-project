#!/bin/bash

# setup_automation.sh
# Script to help set up automated solar flare prediction updates

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp/solar_flare_logs"

echo "🌟 Solar Flare Prediction Automation Setup"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "Log directory: $LOG_DIR"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to show current cron jobs
show_current_cron() {
    echo "📋 Current cron jobs:"
    crontab -l 2>/dev/null || echo "   No cron jobs found"
    echo ""
}

# Function to suggest cron job entries
suggest_cron_entries() {
    echo "💡 Suggested cron job entries:"
    echo ""
    echo "# Run predictions every hour"
    echo "0 * * * * cd '$SCRIPT_DIR' && ./update_predictions.sh >> '$LOG_DIR/predictions.log' 2>&1"
    echo ""
    echo "# Run predictions every 6 hours"
    echo "0 */6 * * * cd '$SCRIPT_DIR' && ./update_predictions.sh >> '$LOG_DIR/predictions.log' 2>&1"
    echo ""
    echo "# Run predictions daily at 6 AM"
    echo "0 6 * * * cd '$SCRIPT_DIR' && ./update_predictions.sh >> '$LOG_DIR/predictions.log' 2>&1"
    echo ""
    echo "# Run system health check every 30 minutes"
    echo "*/30 * * * * cd '$SCRIPT_DIR' && python monitor_system.py >> '$LOG_DIR/health.log' 2>&1"
    echo ""
}

# Function to add a cron job
add_cron_job() {
    local schedule="$1"
    local description="$2"
    
    echo "Adding cron job: $description"
    echo "Schedule: $schedule"
    
    # Get current crontab
    current_cron=$(crontab -l 2>/dev/null || echo "")
    
    # Add new job
    new_job="$schedule cd '$SCRIPT_DIR' && ./update_predictions.sh >> '$LOG_DIR/predictions.log' 2>&1"
    
    # Check if job already exists
    if echo "$current_cron" | grep -q "update_predictions.sh"; then
        echo "⚠️  A similar cron job already exists. Please check manually."
        return 1
    fi
    
    # Add the job
    (echo "$current_cron"; echo "# $description"; echo "$new_job") | crontab -
    echo "✅ Cron job added successfully!"
}

# Function to set up log rotation
setup_log_rotation() {
    echo "📝 Setting up log rotation..."
    
    # Create logrotate config
    cat > "/tmp/solar_flare_logrotate" << EOF
$LOG_DIR/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $(whoami) $(whoami)
}
EOF
    
    echo "✅ Log rotation config created at /tmp/solar_flare_logrotate"
    echo "💡 To enable, run: sudo mv /tmp/solar_flare_logrotate /etc/logrotate.d/solar_flare"
}

# Function to test the setup
test_setup() {
    echo "🧪 Testing setup..."
    
    # Test update script
    if [[ -x "./update_predictions.sh" ]]; then
        echo "✅ Update script is executable"
    else
        echo "❌ Update script is not executable or not found"
        return 1
    fi
    
    # Test monitoring script
    if [[ -x "./monitor_system.py" ]]; then
        echo "✅ Monitoring script is executable"
    else
        echo "❌ Monitoring script is not executable or not found"
        return 1
    fi
    
    # Test log directory
    if [[ -d "$LOG_DIR" ]]; then
        echo "✅ Log directory exists"
    else
        echo "❌ Log directory not found"
        return 1
    fi
    
    # Run a quick health check
    echo "🔍 Running health check..."
    python monitor_system.py --json > /dev/null
    echo "✅ Health check passed"
    
    echo "🎉 Setup test completed successfully!"
}

# Main menu
show_menu() {
    echo "🔧 What would you like to do?"
    echo "1) Show current cron jobs"
    echo "2) Show suggested cron entries"
    echo "3) Add hourly prediction updates"
    echo "4) Add 6-hourly prediction updates"
    echo "5) Add daily prediction updates"
    echo "6) Set up log rotation"
    echo "7) Test setup"
    echo "8) Exit"
    echo ""
    read -p "Enter your choice (1-8): " choice
}

# Main execution
main() {
    while true; do
        echo ""
        show_menu
        
        case $choice in
            1)
                show_current_cron
                ;;
            2)
                suggest_cron_entries
                ;;
            3)
                add_cron_job "0 * * * *" "Solar flare predictions - hourly"
                ;;
            4)
                add_cron_job "0 */6 * * *" "Solar flare predictions - every 6 hours"
                ;;
            5)
                add_cron_job "0 6 * * *" "Solar flare predictions - daily at 6 AM"
                ;;
            6)
                setup_log_rotation
                ;;
            7)
                test_setup
                ;;
            8)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Check if running interactively
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ -t 0 ]]; then
        # Interactive mode
        main
    else
        # Non-interactive mode - just show suggestions
        echo "🤖 Non-interactive mode detected"
        suggest_cron_entries
        echo "💡 Run this script interactively for more options: bash setup_automation.sh"
    fi
fi 