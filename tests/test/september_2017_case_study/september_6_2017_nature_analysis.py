#!/usr/bin/env python3
"""
September 6, 2017 X9.3 Solar Flare - EVEREST Model Analysis
===========================================================

Comprehensive analysis of the September 6, 2017 X9.3 solar flare using the EVEREST model
with the proper Nature dataset and utils.py approach.

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

def load_september_6_data():
    """Load data around September 6, 2017 X9.3 flare using Nature dataset"""
    
    print("🚀 Loading September 6, 2017 X9.3 flare data from Nature dataset...")
    
    # Load the full Nature training dataset
    df = pd.read_csv('Nature_data/training_data_M5_72.csv')
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['DATE__OBS'], utc=True)
    
    # Filter to HARPNUM 7115 (AR 2673)
    df_harpnum = df[df['HARPNUM'] == 7115].copy()
    
    # Filter to September 6, 2017 flare period (72h before to 24h after)
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    # Filter time window
    mask = (df_harpnum['timestamp'] >= start_time) & (df_harpnum['timestamp'] <= end_time)
    df_filtered = df_harpnum[mask].copy()
    
    # Sort by timestamp
    df_filtered = df_filtered.sort_values('timestamp').reset_index(drop=True)
    
    print(f"📊 Data Summary:")
    print(f"   • HARPNUM: 7115 (NOAA AR 2673)")
    print(f"   • Time window: {start_time} to {end_time}")
    print(f"   • Total data points: {len(df_filtered)}")
    print(f"   • Flare time: {flare_time}")
    print(f"   • Positive flare labels: {len(df_filtered[df_filtered['Flare'] == 'P'])}")
    print(f"   • Negative flare labels: {len(df_filtered[df_filtered['Flare'] == 'N'])}")
    
    # Check data availability
    if len(df_filtered) == 0:
        raise ValueError("No data found for the specified time window!")
    
    return df_filtered, flare_time

def prepare_sequences(df_filtered):
    """Prepare sequences for EVEREST model using same approach as July 2012"""
    
    print("\n🔧 Preparing sequences for EVEREST model...")
    
    # Define the 9 SHARP features in correct order
    feature_columns = [
        'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE', 
        'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
    ]
    
    print(f"   • Using features: {feature_columns}")
    
    # Extract features and timestamps
    features = df_filtered[feature_columns].values
    timestamps = df_filtered['timestamp'].values
    flare_labels = df_filtered['Flare'].values
    
    # Create sequences (10 timesteps)
    sequence_length = 10
    sequences = []
    sequence_timestamps = []
    sequence_labels = []
    
    for i in range(sequence_length - 1, len(features)):
        # Get sequence of 10 timesteps
        seq = features[i - sequence_length + 1:i + 1]
        sequences.append(seq)
        sequence_timestamps.append(timestamps[i])
        sequence_labels.append(flare_labels[i])
    
    sequences = np.array(sequences)
    sequence_timestamps = np.array(sequence_timestamps)
    
    print(f"   • Created {len(sequences)} sequences")
    print(f"   • Sequence shape: {sequences.shape}")
    print(f"   • Feature range check:")
    for j, feature in enumerate(feature_columns):
        print(f"     - {feature}: {sequences[:,:,j].min():.2e} to {sequences[:,:,j].max():.2e}")
    
    return sequences, sequence_timestamps, sequence_labels

def run_everest_analysis(sequences):
    """Run EVEREST model analysis"""
    
    print("\n🤖 Running EVEREST model analysis...")
    
    # Load EVEREST model
    print("   • Loading EVEREST model...")
    model = RETPlusWrapper(input_shape=(10, 9))
    model.load('tests/model_weights_EVEREST_72h_M5.pt')
    print("   ✅ Model loaded successfully")
    
    # Get device from the model
    device = next(model.model.parameters()).device
    print(f"   • Model is on device: {device}")
    
    # Run predictions
    print("   • Running predictions...")
    model.model.eval()
    
    with torch.no_grad():
        # Convert to tensor and move to correct device
        X_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
        
        # Get all model outputs
        outputs = model.model(X_tensor)
        
        # Extract all components
        logits = outputs['logits'].cpu().numpy().flatten()
        probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
        
        evid = outputs['evid'].cpu().numpy() if outputs['evid'] is not None else None
        gpd = outputs['gpd'].cpu().numpy() if outputs['gpd'] is not None else None
        precursor = outputs['precursor'].cpu().numpy().flatten() if outputs['precursor'] is not None else None
    
    print(f"   ✅ Predictions completed")
    print(f"   • Primary probabilities shape: {probabilities.shape}")
    print(f"   • Probability range: {probabilities.min():.4f} to {probabilities.max():.4f}")
    
    return {
        'primary_probs': probabilities,
        'logits': logits,
        'evidential_outputs': evid,
        'evt_outputs': gpd,
        'precursor_outputs': precursor
    }

def analyze_results(predictions, sequence_timestamps, sequence_labels, flare_time):
    """Analyze EVEREST results"""
    
    print("\n📊 Analyzing EVEREST Results...")
    
    primary_probs = predictions['primary_probs'] * 100  # Convert to percentage
    
    # Convert timestamps to pandas for easier manipulation
    timestamps_pd = pd.to_datetime(sequence_timestamps, utc=True)
    
    # Key statistics
    max_prob = primary_probs.max()
    max_prob_idx = primary_probs.argmax()
    max_prob_time = timestamps_pd[max_prob_idx]
    mean_prob = primary_probs.mean()
    std_prob = primary_probs.std()
    
    # Probability at flare time (closest timestamp)
    flare_time_diff = np.abs(timestamps_pd - flare_time)
    flare_idx = flare_time_diff.argmin()
    prob_at_flare = primary_probs[flare_idx]
    
    # Lead time analysis
    lead_times = {}
    thresholds = [2, 5, 10, 15, 20, 30]
    
    for threshold in thresholds:
        above_threshold = primary_probs >= threshold
        if above_threshold.any():
            first_alert_idx = np.where(above_threshold)[0][0]
            first_alert_time = timestamps_pd[first_alert_idx]
            lead_time_hours = (flare_time - first_alert_time).total_seconds() / 3600
            lead_times[threshold] = lead_time_hours
        else:
            lead_times[threshold] = 0
    
    # Print results
    print(f"\n🎯 SEPTEMBER 6, 2017 X9.3 FLARE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"📈 PRIMARY PROBABILITY ANALYSIS:")
    print(f"   • Maximum probability: {max_prob:.2f}% at {max_prob_time}")
    print(f"   • Mean probability: {mean_prob:.2f} ± {std_prob:.2f}%")
    print(f"   • Probability at flare time: {prob_at_flare:.2f}%")
    print(f"   • Flare time: {flare_time}")
    
    print(f"\n⏰ LEAD TIME PERFORMANCE:")
    for threshold in thresholds:
        if lead_times[threshold] > 0:
            print(f"   • {threshold:2d}% threshold: {lead_times[threshold]:.1f} hours lead time")
        else:
            print(f"   • {threshold:2d}% threshold: No alert (below threshold)")
    
    # Compare to July 2012 X1.4 results
    print(f"\n📊 COMPARISON TO JULY 2012 X1.4:")
    july_2012_max = 15.76  # From previous analysis
    july_2012_mean = 5.03
    july_2012_flare_prob = 15.72
    
    improvement_max = max_prob / july_2012_max
    improvement_mean = mean_prob / july_2012_mean 
    improvement_flare = prob_at_flare / july_2012_flare_prob
    
    print(f"   • Maximum probability: {improvement_max:.2f}x stronger")
    print(f"   • Mean probability: {improvement_mean:.2f}x stronger") 
    print(f"   • Probability at flare: {improvement_flare:.2f}x stronger")
    
    # Population context
    population_optimal_threshold = 46  # From July 2012 analysis
    population_achievement = max_prob / population_optimal_threshold
    print(f"\n🎯 POPULATION PERFORMANCE CONTEXT:")
    print(f"   • Population optimal threshold: {population_optimal_threshold}%")
    print(f"   • September 2017 achievement: {population_achievement:.1%} of optimal")
    
    return {
        'max_prob': max_prob,
        'max_prob_time': max_prob_time,
        'mean_prob': mean_prob,
        'std_prob': std_prob,
        'prob_at_flare': prob_at_flare,
        'lead_times': lead_times,
        'timestamps': timestamps_pd,
        'probabilities': primary_probs
    }

def create_visualization(results, flare_time):
    """Create visualization similar to July 2012 analysis"""
    
    print("\n📊 Creating visualization...")
    
    timestamps = results['timestamps']
    probabilities = results['probabilities']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot probability
    ax.plot(timestamps, probabilities, 'b-', linewidth=2, label='EVEREST Probability')
    
    # Mark flare time
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, 
               label=f'X9.3 Flare ({flare_time.strftime("%Y-%m-%d %H:%M UTC")})')
    
    # Add threshold lines
    thresholds = [2, 5, 10, 20, 46]
    colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred']
    for threshold, color in zip(thresholds, colors):
        ax.axhline(threshold, color=color, linestyle=':', alpha=0.7, 
                  label=f'{threshold}% threshold')
    
    # Mark maximum probability
    max_prob_time = results['max_prob_time']
    max_prob = results['max_prob']
    ax.plot(max_prob_time, max_prob, 'ro', markersize=10, 
            label=f'Peak: {max_prob:.2f}%')
    
    # Formatting
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_ylabel('Flare Probability (%)', fontsize=12)
    ax.set_title('September 6, 2017 X9.3 Solar Flare - EVEREST Analysis\n' +
                f'HARPNUM 7115 (NOAA AR 2673)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('september_6_2017_x93_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Visualization saved as 'september_6_2017_x93_analysis.png'")
    
    plt.show()

def main():
    """Main analysis function"""
    
    print("🌞 SEPTEMBER 6, 2017 X9.3 SOLAR FLARE ANALYSIS")
    print("=" * 60)
    print("Using Nature dataset and EVEREST model")
    print("Largest flare of Solar Cycle 24\n")
    
    try:
        # Load data
        df_filtered, flare_time = load_september_6_data()
        
        # Prepare sequences
        sequences, sequence_timestamps, sequence_labels = prepare_sequences(df_filtered)
        
        # Run EVEREST analysis
        predictions = run_everest_analysis(sequences)
        
        # Analyze results
        results = analyze_results(predictions, sequence_timestamps, sequence_labels, flare_time)
        
        # Create visualization
        create_visualization(results, flare_time)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"🎯 Key Finding: Peak probability of {results['max_prob']:.2f}% for X9.3 flare")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 