#!/usr/bin/env python3
"""
September 6, 2017 X9.3 Solar Flare - EVEREST Model Analysis
===========================================================

Comprehensive analysis of the September 6, 2017 X9.3 solar flare using the EVEREST model.
This was the LARGEST flare of Solar Cycle 24.

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

from solarknowledge_ret_plus import RETPlusWrapper
import torch

def load_september_6_data():
    """Load data around September 6, 2017 X9.3 flare"""
    
    print("🚀 Loading September 6, 2017 X9.3 flare data...")
    
    # Load the SHARP data in nature format (matches July 2012 format)
    df = pd.read_csv('./archive/data/datasets/sharp_noaa_2017_direct/sharp_flare_data_2017_nature_format.csv')
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['DATE__OBS'], utc=True)
    
    # Filter to HARPNUM 7115 (AR 2673)
    df_harpnum = df[df['HARPNUM'] == 7115].copy()
    
    # Filter to September 6, 2017 flare period (72h before to 24h after)
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    period_data = df_harpnum[(df_harpnum['timestamp'] >= start_time) & 
                            (df_harpnum['timestamp'] <= end_time)].copy()
    
    # Handle missing/invalid data (convert "Invalid KeyLink" to NaN)
    for col in period_data.columns:
        if period_data[col].dtype == 'object':
            period_data[col] = pd.to_numeric(period_data[col], errors='coerce')
    
    print(f"✅ Loaded {len(period_data)} data points from {start_time} to {end_time}")
    print(f"   HARPNUM: 7115 (AR 2673)")
    print(f"   Flare time: {flare_time}")
    print(f"   Data format: Nature format (compatible with July 2012)")
    
    return period_data, flare_time

def prepare_sequences(data, sequence_length=10):
    """Prepare sequences for the model with correct 9 features"""
    
    print("🔧 Preparing sequences for EVEREST model...")
    
    # The 9 features used by the model (same order as July 2012)
    feature_columns = [
        'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE',
        'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
    ]
    
    # Check if all features are available
    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        print(f"Available columns: {data.columns.tolist()}")
        return None, None
    
    # Extract features and handle missing values
    features = data[feature_columns].values
    
    # Replace any NaN/inf values
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Simple standardization (same as July 2012 analysis)
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    # Create sequences
    sequences = []
    timestamps = []
    
    for i in range(len(features) - sequence_length + 1):
        seq = features[i:i+sequence_length]
        sequences.append(seq)
        timestamps.append(data.iloc[i+sequence_length-1]['timestamp'])
    
    print(f"✅ Created {len(sequences)} sequences of length {sequence_length}")
    
    return np.array(sequences), timestamps

def analyze_model_outputs(model, sequences, timestamps, flare_time):
    """Comprehensive analysis of all model outputs"""
    
    print("🔍 Running EVEREST model inference...")
    
    model.model.eval()
    
    # Get device from the model
    device = next(model.model.parameters()).device
    print(f"   Model device: {device}")
    
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
    
    print(f"✅ Processed {len(probabilities)} predictions")
    
    # Create comprehensive dataframe
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'logits': logits,
        'probability': probabilities,
    })
    
    # Add evidential uncertainty if available
    if evid is not None:
        results_df['evid_mu'] = evid[:, 0]
        results_df['evid_nu'] = evid[:, 1] 
        results_df['evid_alpha'] = evid[:, 2]
        results_df['evid_beta'] = evid[:, 3]
        
        # Compute epistemic and aleatoric uncertainty
        nu = evid[:, 1]
        alpha = evid[:, 2] 
        beta = evid[:, 3]
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = beta / (nu * (alpha - 1))
        epistemic_uncertainty[alpha <= 1] = np.inf
        
        # Aleatoric uncertainty (data uncertainty) 
        aleatoric_uncertainty = beta / (alpha - 1)
        aleatoric_uncertainty[alpha <= 1] = np.inf
        
        results_df['epistemic_uncertainty'] = epistemic_uncertainty
        results_df['aleatoric_uncertainty'] = aleatoric_uncertainty
        results_df['total_uncertainty'] = epistemic_uncertainty + aleatoric_uncertainty
    
    # Add EVT parameters if available
    if gpd is not None:
        results_df['evt_xi'] = gpd[:, 0]  # Shape parameter
        results_df['evt_sigma'] = gpd[:, 1]  # Scale parameter
        
        # EVT-based tail risk score
        xi = gpd[:, 0]
        sigma = gpd[:, 1]
        results_df['tail_risk'] = np.abs(xi) * sigma
    
    # Add precursor score if available
    if precursor is not None:
        precursor_probs = 1 / (1 + np.exp(-precursor))  # sigmoid
        results_df['precursor_score'] = precursor_probs
    
    # Calculate time to flare
    results_df['hours_to_flare'] = (flare_time - results_df['timestamp']).dt.total_seconds() / 3600
    
    return results_df

def create_comprehensive_visualization(results_df, flare_time):
    """Create comprehensive visualization of all model outputs"""
    
    print("📊 Creating comprehensive visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    
    # Color scheme
    color_prob = '#2E86AB'      # Blue
    color_flare = '#A23B72'     # Purple-red  
    color_evid = '#F18F01'      # Orange
    color_evt = '#C73E1D'       # Red
    color_precursor = '#8E44AD'  # Purple
    
    # 1. Main flare probability
    ax = axes[0]
    ax.plot(results_df['timestamp'], results_df['probability'] * 100, 
           color=color_prob, linewidth=2.5, label='Flare Probability')
    ax.axvline(flare_time, color=color_flare, linestyle='--', linewidth=2, 
              label='X9.3 Flare (12:02 UTC)')
    
    # Add operational thresholds
    ax.axhline(46, color='red', linestyle=':', alpha=0.7, label='Population Optimal (46%)')
    ax.axhline(10, color='orange', linestyle=':', alpha=0.7, label='Conservative (10%)')
    ax.axhline(5, color='yellow', linestyle=':', alpha=0.7, label='Balanced (5%)')
    ax.axhline(2, color='green', linestyle=':', alpha=0.7, label='Sensitive (2%)')
    
    ax.set_ylabel('Flare Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('EVEREST Model Analysis: September 6, 2017 X9.3 Flare\nLargest Flare of Solar Cycle 24', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    
    # 2. Evidential uncertainty
    ax = axes[1]
    if 'epistemic_uncertainty' in results_df.columns:
        mask = np.isfinite(results_df['epistemic_uncertainty']) & np.isfinite(results_df['aleatoric_uncertainty'])
        if mask.sum() > 0:
            ax.plot(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'epistemic_uncertainty'], 
                   color='purple', linewidth=2, label='Epistemic (Model) Uncertainty')
            ax.plot(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'aleatoric_uncertainty'], 
                   color=color_evid, linewidth=2, label='Aleatoric (Data) Uncertainty')
            ax.set_yscale('log')
    
    ax.axvline(flare_time, color=color_flare, linestyle='--', linewidth=2, label='X9.3 Flare')
    ax.set_ylabel('Uncertainty', fontsize=12, fontweight='bold')
    ax.set_title('Evidential Uncertainty Quantification', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. EVT parameters
    ax = axes[2]
    if 'evt_xi' in results_df.columns:
        ax.plot(results_df['timestamp'], results_df['evt_xi'], 
               color=color_evt, linewidth=2, label='Shape Parameter (ξ)')
        ax2 = ax.twinx()
        ax2.plot(results_df['timestamp'], results_df['evt_sigma'], 
                color='darkred', linewidth=2, linestyle='--', label='Scale Parameter (σ)')
        ax2.set_ylabel('Scale Parameter', fontsize=10)
        ax2.legend(loc='upper right')
    
    ax.axvline(flare_time, color=color_flare, linestyle='--', linewidth=2, label='X9.3 Flare')
    ax.set_ylabel('Shape Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Extreme Value Theory Parameters', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # 4. Precursor detection
    ax = axes[3]
    if 'precursor_score' in results_df.columns:
        ax.plot(results_df['timestamp'], results_df['precursor_score'] * 100, 
               color=color_precursor, linewidth=2, label='Precursor Score')
    
    ax.axvline(flare_time, color=color_flare, linestyle='--', linewidth=2, label='X9.3 Flare')
    ax.set_ylabel('Precursor Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Flare Precursor Detection', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # 5. Ensemble summary
    ax = axes[4]
    # Combine multiple signals into ensemble score
    ensemble_score = results_df['probability'].copy()
    
    if 'precursor_score' in results_df.columns:
        # Weight precursor score with main probability
        ensemble_score = 0.7 * results_df['probability'] + 0.3 * results_df['precursor_score']
    
    ax.plot(results_df['timestamp'], ensemble_score * 100, 
           color='black', linewidth=3, label='Ensemble Score')
    ax.fill_between(results_df['timestamp'], 0, ensemble_score * 100, 
                   alpha=0.3, color='lightgray')
    
    ax.axvline(flare_time, color=color_flare, linestyle='--', linewidth=2, label='X9.3 Flare')
    ax.set_ylabel('Ensemble Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Multi-Modal Ensemble Prediction', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Sep-%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('september_6_2017_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('september_6_2017_comprehensive_analysis.pdf', bbox_inches='tight')
    
    print("✅ Saved: september_6_2017_comprehensive_analysis.png")
    print("✅ Saved: september_6_2017_comprehensive_analysis.pdf")
    
    return fig

def analyze_prediction_quality(results_df, flare_time):
    """Analyze prediction quality and compare with July 2012"""
    
    print("\n📊 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    max_prob = results_df['probability'].max()
    max_prob_time = results_df.loc[results_df['probability'].idxmax(), 'timestamp']
    mean_prob = results_df['probability'].mean()
    std_prob = results_df['probability'].std()
    
    # Flare time probability
    flare_mask = (results_df['timestamp'] >= flare_time - timedelta(minutes=30)) & \
                 (results_df['timestamp'] <= flare_time + timedelta(minutes=30))
    flare_prob = results_df[flare_mask]['probability'].max() if flare_mask.sum() > 0 else np.nan
    
    print(f"Maximum Probability: {max_prob:.4f} ({max_prob*100:.2f}%)")
    print(f"Time of Maximum: {max_prob_time}")
    print(f"Mean Probability: {mean_prob:.4f} ± {std_prob:.4f}")
    print(f"Probability at Flare: {flare_prob:.4f} ({flare_prob*100:.2f}%)" if not np.isnan(flare_prob) else "Probability at Flare: N/A")
    
    # Lead time analysis for different thresholds
    thresholds = [0.46, 0.10, 0.05, 0.02]  # Population optimal, conservative, balanced, sensitive
    threshold_names = ["Population Optimal (46%)", "Conservative (10%)", "Balanced (5%)", "Sensitive (2%)"]
    
    print(f"\nLead Time Analysis:")
    print(f"{'Threshold':<25} {'First Alert':<20} {'Lead Time':<15}")
    print(f"{'-'*60}")
    
    for thresh, name in zip(thresholds, threshold_names):
        alert_mask = results_df['probability'] >= thresh
        if alert_mask.sum() > 0:
            first_alert_time = results_df[alert_mask]['timestamp'].min()
            lead_hours = (flare_time - first_alert_time).total_seconds() / 3600
            print(f"{name:<25} {first_alert_time.strftime('%Y-%m-%d %H:%M'):<20} {lead_hours:.1f}h")
        else:
            print(f"{name:<25} {'No Alert':<20} {'N/A':<15}")
    
    # Population context analysis
    population_optimal = 0.46
    achievement_ratio = max_prob / population_optimal
    
    print(f"\nPopulation Context:")
    print(f"Population Optimal Threshold: {population_optimal:.2f} ({population_optimal*100:.0f}%)")
    print(f"Achievement Ratio: {achievement_ratio:.2f} ({achievement_ratio*100:.1f}% of optimal)")
    
    # Compare with July 2012 X1.4 baseline
    july_max_prob = 0.1576  # From July 2012 analysis
    july_achievement = 0.34  # 34% of population optimum
    
    print(f"\nComparison with July 2012 X1.4:")
    print(f"July 2012 Max Probability: {july_max_prob:.4f} ({july_max_prob*100:.2f}%)")
    print(f"September 2017 Max Probability: {max_prob:.4f} ({max_prob*100:.2f}%)")
    print(f"Improvement Factor: {max_prob/july_max_prob:.2f}×")
    print(f"July 2012 Achievement: {july_achievement:.2f} ({july_achievement*100:.0f}% of optimal)")
    print(f"September 2017 Achievement: {achievement_ratio:.2f} ({achievement_ratio*100:.1f}% of optimal)")
    
    # Uncertainty analysis
    if 'epistemic_uncertainty' in results_df.columns:
        mask = np.isfinite(results_df['epistemic_uncertainty'])
        if mask.sum() > 0:
            avg_epistemic = results_df.loc[mask, 'epistemic_uncertainty'].mean()
            avg_aleatoric = results_df.loc[mask, 'aleatoric_uncertainty'].mean()
            
            print(f"\nUncertainty Analysis:")
            print(f"Average Epistemic Uncertainty: {avg_epistemic:.2f}")
            print(f"Average Aleatoric Uncertainty: {avg_aleatoric:.2f}")
            
            # Correlation between probability and uncertainty
            prob_uncert_corr = results_df['probability'].corr(results_df['epistemic_uncertainty'])
            print(f"Probability-Uncertainty Correlation: {prob_uncert_corr:.3f}")
    
    return {
        'max_probability': max_prob,
        'max_probability_time': max_prob_time,
        'achievement_ratio': achievement_ratio,
        'flare_probability': flare_prob
    }

def main():
    """Main analysis function"""
    
    print("🚀 SEPTEMBER 6, 2017 X9.3 FLARE ANALYSIS")
    print("  EVEREST Model - Largest Flare of Solar Cycle 24")
    print("=" * 80)
    
    try:
        # Load data
        data, flare_time = load_september_6_data()
        
        if len(data) < 10:
            print("❌ Insufficient data points for analysis")
            return
        
        # Prepare sequences
        sequences, timestamps = prepare_sequences(data)
        
        if sequences is None:
            print("❌ Failed to prepare sequences")
            return
        
        # Load EVEREST model
        print("\n🤖 Loading EVEREST model...")
        model = RETPlusWrapper(input_shape=(10, 9))
        model.load('tests/model_weights_EVEREST_72h_M5.pt')
        print("✅ Model loaded successfully")
        
        # Analyze model outputs
        results_df = analyze_model_outputs(model, sequences, timestamps, flare_time)
        
        # Create visualizations
        fig = create_comprehensive_visualization(results_df, flare_time)
        
        # Analyze prediction quality
        performance_metrics = analyze_prediction_quality(results_df, flare_time)
        
        # Save results
        results_file = 'september_6_2017_analysis_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ Results saved to: {results_file}")
        
        print(f"\n🎯 ANALYSIS COMPLETE")
        print(f"   Maximum Probability: {performance_metrics['max_probability']*100:.2f}%")
        print(f"   Achievement Ratio: {performance_metrics['achievement_ratio']*100:.1f}% of population optimal")
        print(f"   Event Significance: Largest flare of Solar Cycle 24")
        
        return results_df, performance_metrics
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results_df, metrics = main() 