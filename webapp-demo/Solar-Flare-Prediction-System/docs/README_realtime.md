# Real-Time Solar Flare Prediction Data Collection System

This system extends your solar flare prediction dataset with real-time SHARP data for operational deployment. It continuously collects the latest magnetogram data and backfills labels once flare outcomes are known.

## 🚀 Features

- **Real-time data collection**: Automatically fetches the latest SHARP parameters from active regions
- **Delayed labeling**: Waits for flare outcomes before labeling samples (avoiding data leakage)
- **Continuous monitoring**: Runs indefinitely with configurable update intervals
- **Automatic cleanup**: Manages disk space by removing old files
- **Robust error handling**: Handles network failures and data unavailability gracefully

## 📁 File Structure

```
Masters Download/
├── download_new.py          # Main dataset builder (now with real-time support)
├── realtime_monitor.py      # Continuous monitoring script
├── README_realtime.md       # This file
└── realtime_data/          # Output directory (created automatically)
    ├── unlabeled_data_YYYYMMDD_HHMMSS.csv  # Raw SHARP data (unlabeled)
    ├── labeled_data_C_YYYYMMDD_HHMMSS.csv  # Labeled data for C-flare task
    ├── labeled_data_M_YYYYMMDD_HHMMSS.csv  # Labeled data for M-flare task
    └── labeled_data_M5_YYYYMMDD_HHMMSS.csv # Labeled data for M5-flare task
```

## 🔧 Usage

### Single Data Collection Run

Collect data once and exit:

```bash
python download_new.py --realtime --output-dir realtime_data --lookback-hours 72 --label-delay-hours 48
```

**Parameters:**
- `--realtime`: Enable real-time mode
- `--output-dir`: Directory to store collected data (default: `realtime_data`)
- `--lookback-hours`: Hours of SHARP data to collect (default: 72)
- `--label-delay-hours`: Hours to wait before labeling samples (default: 48)

### Continuous Monitoring

Run continuously with automatic updates:

```bash
python realtime_monitor.py --interval 12 --output-dir realtime_data
```

**Parameters:**
- `--interval`: Collection interval in hours (default: 12)
- `--output-dir`: Output directory (default: `realtime_data`)
- `--lookback-hours`: Hours of SHARP data to collect (default: 72)
- `--label-delay-hours`: Hours to wait before labeling (default: 48)
- `--cleanup-days`: Days to keep old files (default: 30)
- `--min-free-gb`: Minimum free disk space in GB (default: 5.0)
- `--max-failures`: Max consecutive failures before stopping (default: 5)

### Background Operation

Run as a background service:

```bash
nohup python realtime_monitor.py --interval 6 > monitor.log 2>&1 &
```

## 📊 Data Flow

### 1. Data Collection Phase
```
Current Time: 2025-05-27 12:00 UTC
Lookback: 72 hours
→ Collects SHARP data from 2025-05-24 12:00 to 2025-05-27 12:00
→ Saves as: unlabeled_data_20250527_120000.csv
```

### 2. Labeling Phase
```
Label Delay: 48 hours
Label Cutoff: 2025-05-25 12:00 UTC
→ Labels samples older than cutoff time
→ Checks for flares in next 24h after each sample
→ Saves as: labeled_data_C_20250527_120000.csv (etc.)
```

### 3. Operational Use
```
For prediction at time T:
1. Use samples from T-72h to T (unlabeled)
2. Apply your trained model
3. Generate flare probability predictions
4. Wait 24-48h to verify predictions against actual flares
```

## 🎯 Deployment Strategy

### Development/Testing
```bash
# Test with shorter intervals and recent historical data
python download_new.py --realtime --lookback-hours 24 --label-delay-hours 12
```

### Production Deployment
```bash
# Run every 6 hours with 3-day lookback
python realtime_monitor.py --interval 6 --lookback-hours 72 --label-delay-hours 48
```

### High-Frequency Monitoring
```bash
# Run every 2 hours for rapid response
python realtime_monitor.py --interval 2 --lookback-hours 48 --label-delay-hours 24
```

## ⚠️ Important Considerations

### Data Availability Lag
- **SHARP data**: Usually available within 2-6 hours of observation
- **HARP mappings**: New active regions may take 1-2 weeks to get HARP numbers
- **Flare catalogs**: Real-time flare data available within minutes to hours

### Recommended Settings
- **Lookback period**: 72 hours (captures full AR evolution)
- **Label delay**: 48 hours (ensures flare outcomes are known)
- **Update interval**: 6-12 hours (balances timeliness vs. system load)

### Data Quality
- The system automatically filters for:
  - Central meridian distance ≤ 70°
  - Radial velocity ≤ 3500 m/s
  - QUALITY == 0
  - No NaN values in SHARP parameters

## 🔍 Monitoring and Logs

### Log Files
- `realtime_monitor.log`: Continuous monitoring logs
- Console output: Real-time status updates

### Key Log Messages
```
INFO: Collected 1234 samples from 5 active regions
WARNING: No HARP mapping for AR 14099
ERROR: JSOC query timed out for HARP 13150
INFO: Labeled 567 samples for task C
```

### Health Checks
- Monitor disk space usage
- Check for consecutive failures
- Verify data collection frequency
- Validate label backfilling

## 🛠️ Troubleshooting

### No Data Collected
**Cause**: Very recent ARs without HARP mappings
**Solution**: 
- Use longer lookback periods (72+ hours)
- Check HARP mapping database updates
- Manually add recent AR mappings if known

### JSOC Query Timeouts
**Cause**: Network issues or server overload
**Solution**:
- Automatic retry with timeout handling
- Reduce concurrent queries
- Use cached data when available

### Missing Flare Labels
**Cause**: Flare catalog delays or API issues
**Solution**:
- Increase label delay period
- Check multiple flare data sources (SWPC, DONKI, HEK)
- Manual verification for critical periods

### Disk Space Issues
**Cause**: Accumulating unlabeled files
**Solution**:
- Automatic cleanup of old files
- Adjust retention periods
- Monitor disk usage

## 📈 Performance Optimization

### Reduce Data Volume
```bash
# Shorter lookback for high-frequency updates
python realtime_monitor.py --interval 2 --lookback-hours 24
```

### Batch Processing
```bash
# Longer intervals for batch processing
python realtime_monitor.py --interval 24 --lookback-hours 168  # Weekly
```

### Selective AR Monitoring
Modify the `active_ars` list in `extend_dataset_realtime()` to focus on specific regions of interest.

## 🔮 Integration with ML Models

### Real-Time Prediction Pipeline
1. **Data Collection**: Use this system to gather latest SHARP data
2. **Preprocessing**: Apply same scaling/normalization as training data
3. **Prediction**: Run your trained model on unlabeled samples
4. **Output**: Generate flare probability forecasts
5. **Validation**: Compare predictions with actual outcomes after delay period

### Example Integration
```python
# Load latest unlabeled data
df = pd.read_csv('realtime_data/unlabeled_data_latest.csv')

# Apply preprocessing (use saved scaling parameters)
df_scaled = apply_scaling(df, scales_C_24)

# Generate predictions
predictions = model.predict(df_scaled)

# Output forecasts
forecast_df = pd.DataFrame({
    'DATE_OBS': df['DATE__OBS'],
    'NOAA_AR': df['NOAA_AR'],
    'C_flare_prob': predictions[:, 0],
    'M_flare_prob': predictions[:, 1],
    'M5_flare_prob': predictions[:, 2]
})
```

## 📞 Support

For issues or questions:
1. Check log files for error messages
2. Verify network connectivity to JSOC/SWPC/DONKI
3. Ensure sufficient disk space and permissions
4. Review HARP mapping database for recent ARs

---

**Note**: This system is designed for operational solar flare prediction. Always validate predictions against actual flare outcomes and maintain appropriate safety margins for space weather applications. 