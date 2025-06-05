#!/usr/bin/env python3
"""
September 6, 2017 X9.3 Solar Flare - EVEREST Model Analysis (Proper Utils Approach)
===================================================================================

Comprehensive analysis of the September 6, 2017 X9.3 solar flare using the EVEREST model
with the correct utils.py approach for data loading and model prediction.

This follows the proper approach used in the codebase, using:
- utils.get_training_data() for data loading
- model.predict_proba() for predictions
- Proper data preprocessing and sequence creation

Event Details:
- Date: September 6, 2017
- Peak Time: 12:02 UTC  
- Classification: X9.3 (strongest flare since 2006)
- Active Region: NOAA AR 2673 / HARPNUM 7115
- Historical Significance: Largest flare of Solar Cycle 24

Author: EVEREST Analysis Team
"""

import sys
import os
sys.path.append('/Users/antanaszilinskas/Github/masters-project/models')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import EVEREST model and utilities
from solarknowledge_ret_plus import RETPlusWrapper
import utils
import torch

def analyze_september_6_2017_with_utils():
    """Analyze September 6, 2017 X9.3 flare using proper utils approach"""
    
    print("🌞 SEPTEMBER 6, 2017 X9.3 SOLAR FLARE ANALYSIS")
    print("=" * 60)
    print("Using proper utils.py approach and predict_proba()")
    print("Largest flare of Solar Cycle 24\\n")
    
    # Load model
    print("🤖 Loading EVEREST model...")
    model = RETPlusWrapper(input_shape=(10, 9))
    model.load('tests/model_weights_EVEREST_72h_M5.pt')
    print("   ✅ Model loaded successfully")
    
    # Load data using proper utils approach
    print("\\n📊 Loading data using utils.get_training_data()...")
    X_data, y_data = utils.get_training_data(time_window="72", flare_class="M5")
    print(f"   ✅ Loaded data shape: X={np.array(X_data).shape}, y={len(y_data)}")
    
    # Convert to numpy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    # Load the raw CSV to get timestamps for filtering
    print("\\n🔍 Loading raw data to filter September 6, 2017 period...")
    df_raw = pd.read_csv('Nature_data/training_data_M5_72.csv')
    df_raw['timestamp'] = pd.to_datetime(df_raw['DATE__OBS'], utc=True)
    
    # Define the flare time and analysis window
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    print(f"   • Flare time: {flare_time}")
    print(f"   • Analysis window: {start_time} to {end_time}")
    
    # Filter to the time period and HARPNUM 7115
    period_mask = (df_raw['timestamp'] >= start_time) & (df_raw['timestamp'] <= end_time)
    harpnum_mask = df_raw['HARPNUM'] == 7115
    
    period_data = df_raw[period_mask & harpnum_mask].copy()
    period_data = period_data.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   • Found {len(period_data)} data points for HARPNUM 7115 in analysis period")
    print(f"   • Positive labels: {len(period_data[period_data['Flare'] == 'P'])}")
    print(f"   • Negative labels: {len(period_data[period_data['Flare'] == 'N'])}")
    
    if len(period_data) == 0:
        print("❌ No data found for the specified period!")
        return
    
    # Find corresponding indices in the processed data
    # This is tricky because utils.load_data() does complex processing
    # We need to match the processed sequences to our time period
    
    # For now, let's create sequences for just the September period using utils approach
    print("\\n🔧 Creating sequences for September 2017 period...")
    
    # Temporarily save the period data to a CSV and use utils.load_data
    temp_file = 'temp_september_2017_data.csv'
    period_data.to_csv(temp_file, index=False)
    
    try:
        # Use utils.load_data to process the September data
        X_sept, y_sept, _ = utils.load_data(
            datafile=temp_file,
            flare_label="M5",
            series_len=10,
            start_feature=4,  # Based on utils.py start_feature
            n_features=9,     # 9 SHARP features
            mask_value=0
        )
        
        print(f"   ✅ Created {len(X_sept)} sequences for September 2017")
        print(f"   • Sequence shape: {X_sept.shape}")
        
        # Get timestamps for each sequence
        # The last timestamp of each 10-step sequence
        sequence_timestamps = []
        for i in range(len(X_sept)):
            if i + 9 < len(period_data):
                sequence_timestamps.append(period_data.iloc[i + 9]['timestamp'])
            else:
                sequence_timestamps.append(period_data.iloc[-1]['timestamp'])
        
        sequence_timestamps = pd.to_datetime(sequence_timestamps, utc=True)
        
        # Run predictions using proper predict_proba method
        print("\\n🚀 Running predictions using model.predict_proba()...")
        probabilities = model.predict_proba(X_sept)
        probabilities = probabilities.flatten()  # Ensure 1D array
        
        print(f"   ✅ Predictions completed")
        print(f"   • Probability shape: {probabilities.shape}")
        print(f"   • Probability range: {probabilities.min():.4f} to {probabilities.max():.4f}")
        
        # Analyze results
        analyze_september_results(probabilities, sequence_timestamps, flare_time)
        
        # Create visualization
        create_september_visualization(probabilities, sequence_timestamps, flare_time)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def analyze_september_results(probabilities, timestamps, flare_time):
    """Analyze the September 2017 results"""
    
    # Convert to percentage
    probs_pct = probabilities * 100
    
    # Key statistics
    max_prob = probs_pct.max()
    max_prob_idx = probs_pct.argmax()
    max_prob_time = timestamps[max_prob_idx]
    mean_prob = probs_pct.mean()
    std_prob = probs_pct.std()
    
    # Probability at flare time (closest timestamp)
    flare_time_diff = np.abs(timestamps - flare_time)
    flare_idx = flare_time_diff.argmin()
    prob_at_flare = probs_pct[flare_idx]
    closest_time = timestamps[flare_idx]
    time_diff_minutes = flare_time_diff.iloc[flare_idx].total_seconds() / 60
    
    # Lead time analysis
    lead_times = {}
    thresholds = [1, 2, 5, 10, 15, 20, 30, 46]
    
    for threshold in thresholds:
        above_threshold = probs_pct >= threshold
        if above_threshold.any():
            first_alert_idx = np.where(above_threshold)[0][0]
            first_alert_time = timestamps[first_alert_idx]
            lead_time_hours = (flare_time - first_alert_time).total_seconds() / 3600
            lead_times[threshold] = lead_time_hours
        else:
            lead_times[threshold] = 0
    
    # Print comprehensive results
    print(f"\\n🎯 SEPTEMBER 6, 2017 X9.3 FLARE ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"📈 PRIMARY PROBABILITY ANALYSIS:")
    print(f"   • Maximum probability: {max_prob:.2f}% at {max_prob_time}")
    print(f"   • Mean probability: {mean_prob:.2f} ± {std_prob:.2f}%")
    print(f"   • Probability at flare time: {prob_at_flare:.2f}%")
    print(f"   • Closest timestamp to flare: {closest_time} ({time_diff_minutes:.1f} min diff)")
    print(f"   • Flare time: {flare_time}")
    
    print(f"\\n⏰ LEAD TIME PERFORMANCE:")
    for threshold in thresholds:
        if lead_times[threshold] > 0:
            print(f"   • {threshold:2d}% threshold: {lead_times[threshold]:.1f} hours lead time")
        else:
            print(f"   • {threshold:2d}% threshold: No alert (below threshold)")
    
    # Compare to July 2012 X1.4 results
    print(f"\\n📊 COMPARISON TO JULY 2012 X1.4:")
    july_2012_max = 15.76  # From previous analysis
    july_2012_mean = 5.03
    july_2012_flare_prob = 15.72
    
    improvement_max = max_prob / july_2012_max
    improvement_mean = mean_prob / july_2012_mean 
    improvement_flare = prob_at_flare / july_2012_flare_prob
    
    print(f"   • Maximum probability: {improvement_max:.2f}x (July: {july_2012_max:.2f}% vs Sept: {max_prob:.2f}%)")
    print(f"   • Mean probability: {improvement_mean:.2f}x (July: {july_2012_mean:.2f}% vs Sept: {mean_prob:.2f}%)")
    print(f"   • Probability at flare: {improvement_flare:.2f}x (July: {july_2012_flare_prob:.2f}% vs Sept: {prob_at_flare:.2f}%)")
    
    # Population context
    population_optimal_threshold = 46  # From July 2012 analysis
    population_achievement = max_prob / population_optimal_threshold
    print(f"\\n🎯 POPULATION PERFORMANCE CONTEXT:")
    print(f"   • Population optimal threshold: {population_optimal_threshold}%")
    print(f"   • September 2017 achievement: {population_achievement:.1%} of optimal")
    print(f"   • July 2012 achievement: {july_2012_max/population_optimal_threshold:.1%} of optimal")
    
    # Event magnitude context
    print(f"\\n⚡ EVENT MAGNITUDE CONTEXT:")
    print(f"   • July 2012: X1.4 flare")
    print(f"   • September 2017: X9.3 flare (6.6× stronger)")
    print(f"   • Predictability ratio: {improvement_max:.2f}x despite being 6.6x stronger")
    
    return {
        'max_prob': max_prob,
        'max_prob_time': max_prob_time,
        'mean_prob': mean_prob,
        'std_prob': std_prob,
        'prob_at_flare': prob_at_flare,
        'lead_times': lead_times,
        'timestamps': timestamps,
        'probabilities': probs_pct
    }

def create_september_visualization(probabilities, timestamps, flare_time):
    """Create comprehensive visualization"""
    
    print("\\n📊 Creating visualization...")
    
    probs_pct = probabilities * 100
    
    # Create figure with better layout
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Plot probability with improved styling
    ax.plot(timestamps, probs_pct, 'b-', linewidth=2.5, label='EVEREST Probability', alpha=0.8)
    
    # Mark flare time with better visibility
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=3, 
               label=f'X9.3 Flare\\n{flare_time.strftime("%Y-%m-%d %H:%M UTC")}', alpha=0.9)
    
    # Add threshold lines with clear labeling
    thresholds = [1, 2, 5, 10, 20, 46]
    colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
    alphas = [0.6, 0.6, 0.7, 0.7, 0.8, 0.9]
    
    for threshold, color, alpha in zip(thresholds, colors, alphas):
        ax.axhline(threshold, color=color, linestyle=':', alpha=alpha, linewidth=1.5,
                  label=f'{threshold}% threshold')
    
    # Mark maximum probability
    max_prob = probs_pct.max()
    max_prob_idx = probs_pct.argmax()
    max_prob_time = timestamps[max_prob_idx]
    ax.plot(max_prob_time, max_prob, 'ro', markersize=12, 
            label=f'Peak: {max_prob:.2f}%\\n{max_prob_time.strftime("%m-%d %H:%M")}',
            markeredgecolor='darkred', markeredgewidth=2)
    
    # Annotations for key events
    lead_time_hours = (flare_time - max_prob_time).total_seconds() / 3600
    ax.annotate(f'Lead time: {lead_time_hours:.1f}h', 
                xy=(max_prob_time, max_prob), xytext=(max_prob_time, max_prob + 2),
                arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7),
                fontsize=10, ha='center', color='darkred', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Time (UTC)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flare Probability (%)', fontsize=14, fontweight='bold')
    ax.set_title('September 6, 2017 X9.3 Solar Flare - EVEREST Analysis\\n' +
                f'HARPNUM 7115 (NOAA AR 2673) - Largest Flare of Solar Cycle 24', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Format x-axis with better time labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # Set y-axis limits for better visibility
    ax.set_ylim(-1, max(max_prob + 5, 50))
    
    plt.tight_layout()
    
    # Save figure with high quality
    filename = 'september_6_2017_x93_proper_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Visualization saved as '{filename}'")
    
    plt.show()

def main():
    """Main analysis function"""
    
    try:
        analyze_september_6_2017_with_utils()
        print(f"\\n✅ Analysis completed successfully!")
        print(f"🎯 Key insight: Used proper utils.py approach with model.predict_proba()")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 