# Solar Flare Model Deployment System

This directory contains the production deployment system for the Everest Solar Flare Prediction models. The system automatically runs predictions on all available model combinations and generates JSON data for the web UI.

## 🚀 Quick Start

### One-time Setup
1. Ensure you're in the `models` directory
2. Make sure all model files are present in the `models/` subdirectory
3. Verify that test data is available in the `../data/` directory

### Running Predictions

#### Manual Update
```bash
# Run predictions once
python deploy_predictions_simple.py --update-mode

# Or use the automated script
./update_predictions.sh
```

#### Automated Updates
```bash
# Use the interactive setup script (recommended)
./setup_automation.sh

# Or manually set up a cron job
crontab -e

# Add this line to run every hour:
0 * * * * cd /path/to/Solar-Flare-Prediction-System/models && ./update_predictions.sh >> /tmp/solar_flare_logs/predictions.log 2>&1

# Or run every 6 hours:
0 */6 * * * cd /path/to/Solar-Flare-Prediction-System/models && ./update_predictions.sh >> /tmp/solar_flare_logs/predictions.log 2>&1

# Add health monitoring every 30 minutes:
*/30 * * * * cd /path/to/Solar-Flare-Prediction-System/models && python monitor_system.py >> /tmp/solar_flare_logs/health.log 2>&1
```

## 📁 File Structure

```
models/
├── deploy_predictions_simple.py    # Main prediction script
├── update_predictions.sh           # Automated update script
├── solarknowledge_ret_plus.py     # Model wrapper
├── utils.py                        # Data utilities
├── models/                         # Model weights directory
│   ├── EVEREST-v1.3-C-24h/
│   ├── EVEREST-v1.3-C-48h/
│   ├── EVEREST-v1.3-C-72h/
│   └── ...
└── ../data/                        # Test data directory
    ├── testing_data_C_24.csv
    ├── testing_data_C_48.csv
    └── ...
```

## 🔧 System Components

### 1. Model Deployment Script (`deploy_predictions_simple.py`)

**Purpose**: Runs predictions on all available model combinations and generates comprehensive performance metrics.

**Features**:
- Automatically discovers latest model versions
- Loads and processes test data
- Calculates comprehensive metrics (accuracy, TSS, HSS, etc.)
- Generates JSON output for web UI
- Handles errors gracefully

**Usage**:
```bash
python deploy_predictions_simple.py [options]

Options:
  --output-dir DIR     Output directory (default: ../../src/data)
  --model-dir DIR      Model directory (default: models)
  --update-mode        Use fixed output filename for periodic updates
```

### 2. Update Script (`update_predictions.sh`)

**Purpose**: Automated wrapper script for periodic updates.

**Features**:
- Validates environment and dependencies
- Runs prediction script
- Provides detailed logging
- Checks output file generation
- Returns appropriate exit codes

### 3. System Monitoring (`monitor_system.py`)

**Purpose**: Health monitoring and status checking for the entire system.

**Features**:
- Checks model availability and versions
- Validates data file freshness
- Monitors prediction output status
- Provides detailed health reports
- Supports JSON output for automation

**Usage**:
```bash
python monitor_system.py [options]

Options:
  --verbose, -v        Show detailed information
  --json              Output as JSON
  --watch SECONDS     Watch mode - repeat every N seconds
```

### 4. Automation Setup (`setup_automation.sh`)

**Purpose**: Interactive script to help set up automated prediction updates.

**Features**:
- Guided cron job setup
- Log rotation configuration
- System testing and validation
- Multiple scheduling options

**Usage**:
```bash
./setup_automation.sh    # Interactive mode
```

### 5. Model Performance Panel (Web UI)

**Location**: `../../src/components/forecast/ModelPerformancePanel.tsx`

**Features**:
- Real-time loading of prediction results
- Auto-refresh functionality
- Grid and detailed view modes
- Performance metrics visualization
- Color-coded performance indicators
- Manual refresh controls
- Responsive design

## 📊 Output Files

The system generates two JSON files in `../../src/data/`:

### 1. `latest_predictions.json` (Detailed)
- Complete prediction results
- Individual sample predictions
- Full metadata
- ~13KB file size

### 2. `latest_predictions_compact.json` (UI-optimized)
- Summary statistics only
- Performance metrics
- Model metadata
- ~10KB file size (used by web UI)

## 📈 Metrics Explained

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Skill Scores
- **TSS (True Skill Statistic)**: Recall + Specificity - 1
  - Range: [-1, 1], where 1 is perfect, 0 is no skill
- **HSS (Heidke Skill Score)**: Measures accuracy relative to random chance
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Performance Interpretation
- **TSS ≥ 0.5**: Excellent performance (green)
- **TSS ≥ 0.2**: Good performance (yellow)
- **TSS < 0.2**: Poor performance (red)

## 🔄 Model Combinations

The system tests all available combinations of:
- **Flare Classes**: C, M, M5
- **Time Horizons**: 24h, 48h, 72h
- **Model Versions**: Automatically selects latest available

Current models (as of last update):
- C-class: v1.3 (24h, 48h, 72h)
- M-class: v1.3 (24h, 48h), v1.2 (72h)
- M5-class: v1.3 (48h, 72h), v1.3 (24h - has compatibility issues)

## 🐛 Troubleshooting

### Common Issues

1. **"Model weights not found"**
   - Check that model directories exist in `models/`
   - Verify model file names match expected pattern
   - Run: `python monitor_system.py --verbose` to see all models

2. **"Data file not found"**
   - Ensure test data files exist in `../data/`
   - Check file naming convention: `testing_data_{class}_{horizon}.csv`
   - Run: `python monitor_system.py --verbose` to see all data files

3. **"Size mismatch" errors**
   - Some models may have different input shapes
   - These are logged as warnings and skipped
   - Check model compatibility with current data format

4. **Permission denied on scripts**
   - Run: `chmod +x update_predictions.sh setup_automation.sh monitor_system.py`

5. **Web UI not loading predictions**
   - Check if JSON files exist: `ls -la ../../src/data/latest_predictions*.json`
   - Verify file permissions and web server access
   - Check browser console for fetch errors

### Monitoring and Debugging

- **System Health**: `python monitor_system.py --verbose`
- **Watch Mode**: `python monitor_system.py --watch 30` (check every 30 seconds)
- **JSON Output**: `python monitor_system.py --json` (for automation)
- **Log Files**: Check `/tmp/solar_flare_logs/` for automated run logs
- **File Timestamps**: Verify updates with `ls -la ../../src/data/`

### Performance Issues

- **Slow Predictions**: Check model loading times and data processing
- **Memory Issues**: Monitor system resources during prediction runs
- **Network Issues**: Verify web UI can access prediction JSON files

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Data Integration**
   - Automatic data fetching from solar observatories
   - Live prediction updates

2. **Model Versioning**
   - Automatic model updates
   - A/B testing capabilities

3. **Performance Monitoring**
   - Historical performance tracking
   - Drift detection

4. **Alert System**
   - High-confidence flare predictions
   - Performance degradation alerts

### Extending the System

To add new models:
1. Place model weights in `models/EVEREST-v{version}-{class}-{horizon}h/`
2. Ensure model follows the expected input format
3. Run update script to include in predictions

To modify metrics:
1. Update `calculate_metrics()` in `deploy_predictions_simple.py`
2. Update UI component to display new metrics

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log outputs for error details
3. Verify all dependencies are installed
4. Ensure data and model files are accessible

## 🏆 Performance Summary

Current system performance (last update):
- **Models Tested**: 8/9 successful
- **Success Rate**: 88.9%
- **Mean Accuracy**: 0.500
- **Mean TSS**: -0.750

*Note: Performance metrics are based on test data and may not reflect real-world performance.* 