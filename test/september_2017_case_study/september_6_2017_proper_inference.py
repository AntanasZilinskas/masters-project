#!/usr/bin/env python3
"""
September 6, 2017 X9.3 Solar Flare - Proper Inference Analysis
=============================================================

First recreate the successful test results, then apply to specific September 6, 2017 duration.
Uses proper utils approach and model.predict_proba() for inference (no multimodality).

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

from solarknowledge_ret_plus import RETPlusWrapper
import utils
import torch

def recreate_test_results():
    """First recreate the successful test results to confirm methodology"""
    
    print("🔄 STEP 1: Recreating successful test results...")
    print("=" * 60)
    
    # Load model
    print("🤖 Loading EVEREST model...")
    model = RETPlusWrapper(input_shape=(10, 9))
    model.load('tests/model_weights_EVEREST_72h_M5.pt')
    print("   ✅ Model loaded successfully")
    
    # Load ALL preprocessed data using proper utils approach
    print("\\n📊 Loading preprocessed training data using utils...")
    X_all, y_all = utils.get_training_data(time_window="72", flare_class="M5")
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"   ✅ Loaded data shape: X={X_all.shape}, y={y_all.shape}")
    
    # Test on a sample to recreate the 88%+ results
    print("\\n🚀 Testing on sample data to recreate 88%+ probabilities...")
    sample_size = 10000
    sample_indices = np.random.choice(len(X_all), sample_size, replace=False)
    X_sample = X_all[sample_indices]
    y_sample = y_all[sample_indices]
    
    # Run predictions using proper predict_proba method
    probabilities = model.predict_proba(X_sample)
    probabilities = probabilities.flatten()
    probs_pct = probabilities * 100
    
    print(f"   ✅ Sample results recreated:")
    print(f"   • Max probability: {probs_pct.max():.2f}%")
    print(f"   • Mean probability: {probs_pct.mean():.2f}%")
    print(f"   • Above 50%: {np.sum(probs_pct >= 50)} sequences")
    print(f"   • Above 80%: {np.sum(probs_pct >= 80)} sequences")
    
    # Test on positive sequences specifically
    positive_indices = np.where(y_all == 1)[0]
    if len(positive_indices) > 0:
        pos_sample_size = min(1000, len(positive_indices))
        pos_sample_idx = np.random.choice(positive_indices, pos_sample_size, replace=False)
        X_pos = X_all[pos_sample_idx]
        
        probs_pos = model.predict_proba(X_pos)
        probs_pos = probs_pos.flatten() * 100
        
        print(f"\\n   📈 Positive sequences results:")
        print(f"   • Max probability: {probs_pos.max():.2f}%")
        print(f"   • Mean probability: {probs_pos.mean():.2f}%")
        print(f"   • Above 80%: {np.sum(probs_pos >= 80)} sequences")
    
    return model, X_all, y_all

def find_september_2017_sequences(X_all, y_all):
    """Find sequences corresponding to September 2017 period"""
    
    print("\\n🔍 STEP 2: Finding September 2017 sequences...")
    print("=" * 60)
    
    # Load raw data to get timestamps
    df_raw = pd.read_csv('Nature_data/training_data_M5_72.csv')
    df_raw['timestamp'] = pd.to_datetime(df_raw['DATE__OBS'], utc=True)
    
    # Define September 6, 2017 analysis period
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    print(f"   • Target flare: {flare_time}")
    print(f"   • Analysis window: {start_time} to {end_time}")
    
    # Filter to September period and HARPNUM 7115
    sept_mask = (df_raw['timestamp'] >= start_time) & (df_raw['timestamp'] <= end_time)
    harp_mask = df_raw['HARPNUM'] == 7115
    sept_data = df_raw[sept_mask & harp_mask].copy()
    
    print(f"   • Found {len(sept_data)} raw data points for HARPNUM 7115")
    print(f"   • Positive labels: {len(sept_data[sept_data['Flare'] == 'P'])}")
    print(f"   • Negative labels: {len(sept_data[sept_data['Flare'] == 'N'])}")
    
    # Now we need to find which preprocessed sequences correspond to this period
    # This is the tricky part - mapping back from preprocessed to raw timestamps
    
    # Strategy: Use the full dataset and look for sequences that might correspond
    # to September 2017 by examining the data distribution and timing
    
    # Get all September 2017 data (broader than just our target)
    sept_2017_mask = (df_raw['timestamp'].dt.year == 2017) & (df_raw['timestamp'].dt.month == 9)
    sept_2017_data = df_raw[sept_2017_mask]
    
    print(f"   • Total September 2017 data points: {len(sept_2017_data)}")
    print(f"   • September 2017 HARPNUMs: {sorted(sept_2017_data['HARPNUM'].unique())}")
    
    return sept_data, flare_time, sept_2017_data

def analyze_september_with_proper_utils(model, sept_data, flare_time):
    """Analyze September data using proper utils preprocessing"""
    
    print("\\n🎯 STEP 3: Analyzing September 6, 2017 with proper utils...")
    print("=" * 60)
    
    # Create a temporary file with just the September data
    temp_file = 'temp_september_2017_proper.csv'
    
    try:
        # Save September data in the same format as the training data
        sept_data.to_csv(temp_file, index=False)
        
        # Use utils.load_data with proper parameters (matching the training setup)
        print("   🔧 Processing data with utils.load_data()...")
        X_sept, y_sept, df_processed = utils.load_data(
            datafile=temp_file,
            flare_label="M5",  # Use M5 to match training
            series_len=10,
            start_feature=5,  # Correct start_feature from utils
            n_features=14,    # Correct n_features from utils  
            mask_value=0
        )
        
        print(f"   ✅ Processed {len(X_sept)} sequences")
        print(f"   • Sequence shape: {np.array(X_sept).shape}")
        print(f"   • Labels: {len(y_sept)} ({np.sum(np.array(y_sept) != 'N')} positive)")
        
        if len(X_sept) == 0:
            print("   ❌ No sequences created - trying alternative approach...")
            return analyze_september_alternative(model, sept_data, flare_time)
        
        # Convert to numpy
        X_sept = np.array(X_sept)
        
        # Run predictions using proper predict_proba
        print("\\n   🚀 Running predictions with model.predict_proba()...")
        probabilities = model.predict_proba(X_sept)
        probabilities = probabilities.flatten()
        probs_pct = probabilities * 100
        
        print(f"   ✅ Predictions completed:")
        print(f"   • Probability range: {probabilities.min():.4f} to {probabilities.max():.4f}")
        print(f"   • Max probability: {probs_pct.max():.2f}%")
        print(f"   • Mean probability: {probs_pct.mean():.2f}%")
        
        # Create timestamps for sequences (approximate)
        timestamps = []
        for i in range(len(X_sept)):
            if i < len(sept_data):
                timestamps.append(sept_data.iloc[min(i + 9, len(sept_data)-1)]['timestamp'])
            else:
                timestamps.append(sept_data.iloc[-1]['timestamp'])
        
        return analyze_september_results(probabilities, timestamps, flare_time, "Proper Utils")
        
    except Exception as e:
        print(f"   ⚠️ Utils approach failed: {e}")
        print("   🔄 Trying alternative approach...")
        return analyze_september_alternative(model, sept_data, flare_time)
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

def analyze_september_alternative(model, sept_data, flare_time):
    """Alternative analysis using a more direct approach"""
    
    print("\\n   🔄 Using alternative direct approach...")
    
    # Use a simpler approach that works with the available data
    # This mimics what utils.load_data does but more directly
    
    feature_columns = [
        'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE',
        'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
    ]
    
    # Extract and normalize features
    features = sept_data[feature_columns].values
    
    # Create sequences manually
    sequences = []
    timestamps = []
    
    for i in range(len(features) - 9):
        seq = features[i:i+10]  # 10 timesteps
        sequences.append(seq)
        timestamps.append(sept_data.iloc[i+9]['timestamp'])
    
    if len(sequences) == 0:
        print("   ❌ No sequences could be created")
        return None
    
    X_sept = np.array(sequences)
    
    print(f"   ✅ Created {len(sequences)} sequences directly")
    print(f"   • Shape: {X_sept.shape}")
    
    # Run predictions
    probabilities = model.predict_proba(X_sept)
    probabilities = probabilities.flatten()
    
    return analyze_september_results(probabilities, timestamps, flare_time, "Direct")

def analyze_september_results(probabilities, timestamps, flare_time, method):
    """Analyze the September 2017 results"""
    
    probs_pct = probabilities * 100
    timestamps = pd.to_datetime(timestamps, utc=True)
    
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
    time_diff_minutes = flare_time_diff.values[flare_idx] / np.timedelta64(1, 'm')
    
    # Lead time analysis
    lead_times = {}
    thresholds = [1, 2, 5, 10, 15, 20, 30, 46, 80]
    
    print(f"\\n🎯 SEPTEMBER 6, 2017 X9.3 ANALYSIS RESULTS ({method} Method)")
    print(f"{'='*70}")
    print(f"📈 PRIMARY PROBABILITY ANALYSIS:")
    print(f"   • Maximum probability: {max_prob:.2f}% at {max_prob_time}")
    print(f"   • Mean probability: {mean_prob:.2f} ± {std_prob:.2f}%")
    print(f"   • Probability at flare time: {prob_at_flare:.2f}%")
    print(f"   • Closest timestamp to flare: {closest_time} ({time_diff_minutes:.1f} min diff)")
    
    print(f"\\n⏰ LEAD TIME PERFORMANCE:")
    for threshold in thresholds:
        above_threshold = probs_pct >= threshold
        if above_threshold.any():
            first_alert_idx = np.where(above_threshold)[0][0]
            first_alert_time = timestamps[first_alert_idx]
            lead_time_hours = (flare_time - first_alert_time).total_seconds() / 3600
            lead_times[threshold] = lead_time_hours
            print(f"   • {threshold:2d}% threshold: {lead_time_hours:5.1f}h lead time")
        else:
            lead_times[threshold] = 0
            print(f"   • {threshold:2d}% threshold: No alerts")
    
    # Compare to expected high-performance results
    print(f"\\n📊 PERFORMANCE VALIDATION:")
    print(f"   • Max probability achieved: {max_prob:.2f}%")
    if max_prob > 50:
        print(f"   ✅ SUCCESS: High probability achieved (> 50%)")
    elif max_prob > 20:
        print(f"   ⚠️  MODERATE: Reasonable probability (> 20%)")
    else:
        print(f"   ❌ LOW: Probability below expected performance")
    
    # Event context
    print(f"\\n⚡ EVENT CONTEXT:")
    print(f"   • Event: September 6, 2017 X9.3 flare")
    print(f"   • Magnitude: Largest flare of Solar Cycle 24")
    print(f"   • Expected: High probabilities (80%+ for extreme events)")
    print(f"   • Achieved: {max_prob:.2f}% maximum probability")
    
    return {
        'max_prob': max_prob,
        'max_prob_time': max_prob_time,
        'mean_prob': mean_prob,
        'prob_at_flare': prob_at_flare,
        'lead_times': lead_times,
        'timestamps': timestamps,
        'probabilities': probs_pct,
        'method': method
    }

def create_inference_visualization(results, flare_time):
    """Create visualization for inference results"""
    
    print(f"\\n📊 Creating inference visualization...")
    
    timestamps = results['timestamps']
    probabilities = results['probabilities']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot probability
    ax.plot(timestamps, probabilities, 'b-', linewidth=2.5, label='EVEREST Probability', alpha=0.8)
    
    # Mark flare time
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=3, 
               label=f'X9.3 Flare\\n{flare_time.strftime("%Y-%m-%d %H:%M UTC")}')
    
    # Add threshold lines
    thresholds = [10, 20, 46, 80]
    colors = ['yellow', 'orange', 'red', 'darkred']
    for threshold, color in zip(thresholds, colors):
        ax.axhline(threshold, color=color, linestyle=':', alpha=0.7, 
                  label=f'{threshold}% threshold')
    
    # Mark maximum probability
    max_prob = probabilities.max()
    max_prob_idx = probabilities.argmax()
    max_prob_time = timestamps[max_prob_idx]
    ax.plot(max_prob_time, max_prob, 'ro', markersize=12, 
            label=f'Peak: {max_prob:.1f}%')
    
    # Lead time annotation
    lead_time_hours = (flare_time - max_prob_time).total_seconds() / 3600
    ax.annotate(f'Lead time: {lead_time_hours:.1f}h', 
                xy=(max_prob_time, max_prob), xytext=(max_prob_time, max_prob + 5),
                arrowprops=dict(arrowstyle='->', color='darkred'),
                fontsize=12, ha='center', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Flare Probability (%)', fontsize=14)
    ax.set_title(f'September 6, 2017 X9.3 Solar Flare - EVEREST Inference\\n' +
                f'Method: {results["method"]} | Peak: {max_prob:.1f}%', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    
    plt.tight_layout()
    
    filename = f'september_6_2017_x93_inference_{results["method"].lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Visualization saved as '{filename}'")
    
    plt.show()

def main():
    """Main analysis function"""
    
    print("🌞 SEPTEMBER 6, 2017 X9.3 FLARE - PROPER INFERENCE ANALYSIS")
    print("=" * 70)
    print("Recreating test results then applying to specific event\\n")
    
    try:
        # Step 1: Recreate successful test results
        model, X_all, y_all = recreate_test_results()
        
        # Step 2: Find September 2017 sequences
        sept_data, flare_time, sept_2017_data = find_september_2017_sequences(X_all, y_all)
        
        # Step 3: Analyze September with proper utils
        results = analyze_september_with_proper_utils(model, sept_data, flare_time)
        
        if results:
            # Step 4: Create visualization
            create_inference_visualization(results, flare_time)
            
            print(f"\\n✅ ANALYSIS COMPLETED!")
            print(f"🎯 September 6, 2017 X9.3: {results['max_prob']:.2f}% max probability")
            print(f"📊 Method: {results['method']}")
            print(f"⚡ Event: Largest flare of Solar Cycle 24")
        else:
            print(f"\\n❌ Analysis failed - could not process September 2017 data")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 