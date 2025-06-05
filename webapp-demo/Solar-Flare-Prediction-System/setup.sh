#!/bin/bash
# Solar Flare Prediction System Setup Script

echo "🌟 Setting up Solar Flare Prediction System..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p realtime_data
mkdir -p forecasts
mkdir -p logs

# Set executable permissions
echo "🔧 Setting permissions..."
chmod +x core/download_new.py
chmod +x scripts/realtime_monitor.py
chmod +x examples/example_prediction.py

echo "✅ Setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  Historical datasets:  cd core && python download_new.py --start 2025-04-01 --end 2025-04-30"
echo "  Real-time collection: cd core && python download_new.py --realtime"
echo "  Continuous monitor:   cd scripts && python realtime_monitor.py"
echo ""
echo "📚 See README.md for detailed usage instructions." 