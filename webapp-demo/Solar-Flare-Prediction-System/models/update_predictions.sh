#!/bin/bash

# update_predictions.sh
# Script to periodically update solar flare model predictions
# This script should be run from the models directory

set -e  # Exit on any error

echo "🌟 Starting Solar Flare Prediction Update..."
echo "Timestamp: $(date)"
echo "Working directory: $(pwd)"

# Check if we're in the right directory
if [[ ! -f "deploy_predictions_simple.py" ]]; then
    echo "❌ Error: deploy_predictions_simple.py not found. Please run this script from the models directory."
    exit 1
fi

# Check if the data directory exists
if [[ ! -d "../data" ]]; then
    echo "❌ Error: Data directory not found. Please ensure the data directory exists."
    exit 1
fi

# Check if the output directory exists, create if not
OUTPUT_DIR="../../src/data"
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "📁 Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

echo "🔄 Running model predictions..."

# Run the prediction script
python deploy_predictions_simple.py --update-mode

# Check if the prediction was successful
if [[ $? -eq 0 ]]; then
    echo "✅ Model predictions completed successfully!"
    
    echo "🔄 Generating real-time forecast data..."
    # Generate forecast data for the web UI
    python generate_forecast.py
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Forecast data generated successfully!"
    else
        echo "⚠️  Warning: Forecast generation failed, but predictions were successful"
    fi
    
    # Final success check
    if [[ -f "$OUTPUT_DIR/latest_predictions.json" && -f "$OUTPUT_DIR/latest_predictions_compact.json" ]]; then
        echo "📊 Prediction files generated:"
        echo "  - $OUTPUT_DIR/latest_predictions.json"
        echo "  - $OUTPUT_DIR/latest_predictions_compact.json"
        
        # Check for forecast files too
        if [[ -f "$OUTPUT_DIR/forecast_data.json" && -f "$OUTPUT_DIR/temporal_evolution.json" ]]; then
            echo "  - $OUTPUT_DIR/forecast_data.json"
            echo "  - $OUTPUT_DIR/temporal_evolution.json"
        fi
        
        # Show file sizes
        echo "📏 File sizes:"
        ls -lh "$OUTPUT_DIR/latest_predictions"*.json "$OUTPUT_DIR/forecast_data.json" "$OUTPUT_DIR/temporal_evolution.json" 2>/dev/null | awk '{print "  - " $9 ": " $5}'
        
        # Show timestamp of generation
        echo "🕒 Generated at: $(date -r "$OUTPUT_DIR/latest_predictions.json" '+%Y-%m-%d %H:%M:%S')"
        
        echo "🎉 Update completed successfully!"
    else
        echo "⚠️  Warning: Prediction files were not found after running the script."
        exit 1
    fi
else
    echo "❌ Error: Model prediction script failed"
    exit 1
fi

echo "🌟 Solar Flare Prediction Update Complete!"
echo "Timestamp: $(date)"
echo "==========================================" 