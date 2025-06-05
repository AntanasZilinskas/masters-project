#!/usr/bin/env python3
"""
September 6, 2017 X9.3 Solar Flare - Complete EVEREST Analysis
==============================================================

Complete analysis of September 6, 2017 X9.3 flare using EVEREST model.
Adapted from the successful July 2012 methodology to provide comprehensive results.

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
    
    # Load the data file
    df = pd.read_csv('/Users/antanaszilinskas/Github/masters-project/Nature_data/training_data_M5_72.csv')
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['DATE__OBS'], utc=True)
    
    # Filter to September 6, 2017 flare period (72h before to 24h after)
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    # Filter to HARPNUM 7115 (AR 2673) and time period
    period_data = df[
        (df['timestamp'] >= start_time) & 
        (df['timestamp'] <= end_time) &
        (df['HARPNUM'] == 7115)
    ].copy()
    
    period_data = period_data.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Loaded {len(period_data)} data points from {start_time} to {end_time}")
    print(f"   • HARPNUM: 7115 (NOAA AR 2673)")
    print(f"   • Positive labels: {len(period_data[period_data['Flare'] == 'P'])}")
    print(f"   • Negative labels: {len(period_data[period_data['Flare'] == 'N'])}")
    
    return period_data, flare_time

def prepare_sequences(data, sequence_length=10):
    """Prepare sequences for the model with correct 9 features"""
    
    print("\\n🔧 Preparing sequences for EVEREST model...")
    
    # The 9 features used by the model (from models/utils.py)
    feature_columns = [
        'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE',
        'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
    ]
    
    # Extract features and normalize
    features = data[feature_columns].values
    
    # Simple standardization (matching July 2012 approach)
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    # Create sequences
    sequences = []
    timestamps = []
    
    for i in range(len(features) - sequence_length + 1):
        seq = features[i:i+sequence_length]
        sequences.append(seq)
        timestamps.append(data.iloc[i+sequence_length-1]['timestamp'])
    
    print(f"   ✅ Created {len(sequences)} sequences")
    print(f"   • Sequence shape: {np.array(sequences).shape}")
    
    return np.array(sequences), timestamps

def analyze_model_outputs(model, sequences, timestamps, flare_time):
    """Comprehensive analysis of all model outputs"""
    
    print("\\n🤖 Running EVEREST model analysis...")
    
    model.model.eval()
    
    # Get device from the model
    device = next(model.model.parameters()).device
    print(f"   • Model device: {device}")
    
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
    print(f"   • Probability range: {probabilities.min():.4f} to {probabilities.max():.4f}")
    
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
        epistemic_uncertainty[alpha <= 1] = np.inf  # Undefined when alpha <= 1
        
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

def analyze_prediction_quality(results_df, flare_time):
    """Complete analysis of prediction quality and alert characteristics"""
    
    print("\\n" + "="*70)
    print("🎯 SEPTEMBER 6, 2017 X9.3 FLARE ANALYSIS RESULTS")
    print("="*70)
    
    # Convert probabilities to percentage
    probs_pct = results_df['probability'] * 100
    
    # Basic probability statistics
    max_prob = probs_pct.max()
    max_prob_idx = probs_pct.argmax()
    max_prob_time = results_df.iloc[max_prob_idx]['timestamp']
    mean_prob = probs_pct.mean()
    std_prob = probs_pct.std()
    
    # Probability at flare time (closest timestamp)
    flare_time_diff = np.abs(results_df['timestamp'] - flare_time)
    flare_idx = flare_time_diff.argmin()
    prob_at_flare = probs_pct.iloc[flare_idx]
    closest_time = results_df.iloc[flare_idx]['timestamp']
    time_diff_minutes = flare_time_diff.iloc[flare_idx].total_seconds() / 60
    
    print(f"\\n📈 PRIMARY PROBABILITY ANALYSIS:")
    print(f"   • Maximum probability: {max_prob:.2f}% at {max_prob_time}")
    print(f"   • Mean probability: {mean_prob:.2f} ± {std_prob:.2f}%")
    print(f"   • Probability at flare time: {prob_at_flare:.2f}%")
    print(f"   • Closest timestamp to flare: {closest_time} ({time_diff_minutes:.1f} min diff)")
    print(f"   • Flare time: {flare_time}")
    
    # Lead time analysis
    thresholds = [1, 2, 5, 10, 15, 20, 30, 46]
    lead_times = {}
    
    print(f"\\n⏰ LEAD TIME PERFORMANCE:")
    for threshold in thresholds:
        alerts = results_df[probs_pct >= threshold]
        if len(alerts) > 0:
            first_alert = alerts.iloc[0]
            lead_time = first_alert['hours_to_flare']
            lead_times[threshold] = lead_time
            print(f"   • {threshold:2d}% threshold: {lead_time:5.1f}h lead time ({first_alert['timestamp'].strftime('%m/%d %H:%M')})")
        else:
            lead_times[threshold] = 0
            print(f"   • {threshold:2d}% threshold: No alerts")
    
    # Compare to July 2012 X1.4 results
    print(f"\\n📊 COMPARISON TO JULY 2012 X1.4:")
    july_2012_max = 15.76  # From previous analysis
    july_2012_mean = 5.03
    july_2012_flare_prob = 15.72
    
    improvement_max = max_prob / july_2012_max
    improvement_mean = mean_prob / july_2012_mean 
    improvement_flare = prob_at_flare / july_2012_flare_prob
    
    print(f"   • Maximum probability: {improvement_max:.2f}x (Sept: {max_prob:.2f}% vs July: {july_2012_max:.2f}%)")
    print(f"   • Mean probability: {improvement_mean:.2f}x (Sept: {mean_prob:.2f}% vs July: {july_2012_mean:.2f}%)")
    print(f"   • Probability at flare: {improvement_flare:.2f}x (Sept: {prob_at_flare:.2f}% vs July: {july_2012_flare_prob:.2f}%)")
    
    # Population context
    population_optimal_threshold = 46  # From July 2012 analysis
    september_achievement = max_prob / population_optimal_threshold
    july_achievement = july_2012_max / population_optimal_threshold
    
    print(f"\\n🎯 POPULATION PERFORMANCE CONTEXT:")
    print(f"   • Population optimal threshold: {population_optimal_threshold}%")
    print(f"   • September 2017 achievement: {september_achievement:.1%} of optimal")
    print(f"   • July 2012 achievement: {july_achievement:.1%} of optimal")
    
    # Event magnitude context
    print(f"\\n⚡ EVENT MAGNITUDE vs PREDICTABILITY:")
    print(f"   • July 2012: X1.4 flare → {july_2012_max:.2f}% max probability")
    print(f"   • September 2017: X9.3 flare (6.6× stronger) → {max_prob:.2f}% max probability")
    print(f"   • Predictability scaling: {improvement_max:.2f}x despite 6.6x magnitude increase")
    
    # Multi-modal analysis if available
    if 'total_uncertainty' in results_df.columns:
        print(f"\\n🔬 EVIDENTIAL UNCERTAINTY ANALYSIS:")
        
        # Filter finite values
        mask = np.isfinite(results_df['epistemic_uncertainty']) & np.isfinite(results_df['aleatoric_uncertainty'])
        if mask.sum() > 0:
            epistemic_mean = results_df.loc[mask, 'epistemic_uncertainty'].mean()
            aleatoric_mean = results_df.loc[mask, 'aleatoric_uncertainty'].mean()
            
            print(f"   • Mean epistemic uncertainty: {epistemic_mean:.4f}")
            print(f"   • Mean aleatoric uncertainty: {aleatoric_mean:.4f}")
            
            # Uncertainty correlation with probability
            prob_subset = probs_pct[mask]
            epistemic_subset = results_df.loc[mask, 'epistemic_uncertainty']
            if len(prob_subset) > 1:
                correlation = np.corrcoef(prob_subset, epistemic_subset)[0, 1]
                print(f"   • Probability-epistemic correlation: {correlation:.3f}")
    
    # EVT analysis if available
    if 'tail_risk' in results_df.columns:
        print(f"\\n🌊 EXTREME VALUE THEORY (EVT) ANALYSIS:")
        xi_mean = results_df['evt_xi'].mean()
        sigma_mean = results_df['evt_sigma'].mean()
        tail_risk_mean = results_df['tail_risk'].mean()
        
        print(f"   • Mean shape parameter (ξ): {xi_mean:.3f}")
        print(f"   • Mean scale parameter (σ): {sigma_mean:.3f}")
        print(f"   • Mean tail risk score: {tail_risk_mean:.3f}")
        
        if xi_mean < 0:
            print(f"   • Distribution type: Bounded (finite upper limit)")
        elif xi_mean > 0:
            print(f"   • Distribution type: Heavy-tailed")
        else:
            print(f"   • Distribution type: Exponential")
    
    # Precursor analysis if available
    if 'precursor_score' in results_df.columns:
        print(f"\\n🔍 PRECURSOR DETECTION ANALYSIS:")
        precursor_mean = results_df['precursor_score'].mean()
        precursor_max = results_df['precursor_score'].max()
        
        # Correlation with main probability
        precursor_correlation = np.corrcoef(results_df['probability'], results_df['precursor_score'])[0, 1]
        
        print(f"   • Mean precursor score: {precursor_mean:.3f}")
        print(f"   • Max precursor score: {precursor_max:.3f}")
        print(f"   • Correlation with main probability: {precursor_correlation:.3f}")
    
    return {
        'max_prob': max_prob,
        'max_prob_time': max_prob_time,
        'mean_prob': mean_prob,
        'prob_at_flare': prob_at_flare,
        'lead_times': lead_times,
        'results_df': results_df
    }

def create_comprehensive_visualization(results_df, flare_time):
    """Create comprehensive visualization of all model outputs"""
    
    print("\\n📊 Creating comprehensive visualization...")
    
    # Determine number of subplots based on available data
    n_plots = 2  # Always have probability and logits
    if 'total_uncertainty' in results_df.columns:
        n_plots += 1
    if 'tail_risk' in results_df.columns:
        n_plots += 1
    if 'precursor_score' in results_df.columns:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 5*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 1. Main flare probability with thresholds
    ax = axes[plot_idx]
    probs_pct = results_df['probability'] * 100
    ax.plot(results_df['timestamp'], probs_pct, 'b-', linewidth=2.5, label='EVEREST Probability')
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X9.3 Flare')
    
    # Add threshold lines
    thresholds = [2, 5, 10, 20, 46]
    colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred']
    for threshold, color in zip(thresholds, colors):
        ax.axhline(threshold, color=color, linestyle=':', alpha=0.7, label=f'{threshold}% threshold')
    
    # Mark maximum probability
    max_prob = probs_pct.max()
    max_prob_idx = probs_pct.argmax()
    max_prob_time = results_df.iloc[max_prob_idx]['timestamp']
    ax.plot(max_prob_time, max_prob, 'ro', markersize=10, label=f'Peak: {max_prob:.2f}%')
    
    ax.set_ylabel('Flare Probability (%)', fontsize=12)
    ax.set_title('September 6, 2017 X9.3 Solar Flare - EVEREST Analysis\\nHARPNUM 7115 (NOAA AR 2673)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, max(max_prob + 5, 50))
    plot_idx += 1
    
    # 2. Logits (raw model output)
    ax = axes[plot_idx]
    ax.plot(results_df['timestamp'], results_df['logits'], 'g-', linewidth=2, label='Logits')
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X9.3 Flare')
    ax.set_ylabel('Logits', fontsize=12)
    ax.set_title('Raw Model Logits', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_idx += 1
    
    # 3. Evidential uncertainty if available
    if 'total_uncertainty' in results_df.columns:
        ax = axes[plot_idx]
        
        # Filter out infinite values for plotting
        mask = np.isfinite(results_df['epistemic_uncertainty']) & np.isfinite(results_df['aleatoric_uncertainty'])
        
        if mask.sum() > 0:
            ax.plot(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'epistemic_uncertainty'], 
                   'purple', linewidth=2, label='Epistemic (Model) Uncertainty')
            ax.plot(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'aleatoric_uncertainty'], 
                   'orange', linewidth=2, label='Aleatoric (Data) Uncertainty')
        
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X9.3 Flare')
        ax.set_ylabel('Uncertainty', fontsize=12)
        ax.set_title('Evidential Uncertainty Quantification', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_yscale('log')
        plot_idx += 1
    
    # 4. EVT tail risk if available
    if 'tail_risk' in results_df.columns:
        ax = axes[plot_idx]
        ax.plot(results_df['timestamp'], results_df['tail_risk'], 'brown', linewidth=2, label='Tail Risk Score')
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X9.3 Flare')
        ax.set_ylabel('Tail Risk', fontsize=12)
        ax.set_title('Extreme Value Theory (EVT) Tail Risk', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
    # 5. Precursor score if available
    if 'precursor_score' in results_df.columns:
        ax = axes[plot_idx]
        ax.plot(results_df['timestamp'], results_df['precursor_score'], 'teal', linewidth=2, label='Precursor Score')
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X9.3 Flare')
        ax.set_ylabel('Precursor Probability', fontsize=12)
        ax.set_title('Precursor Activity Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
    
    # Format x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
    axes[-1].set_xlabel('Date (UTC)', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'september_6_2017_x93_complete_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Comprehensive visualization saved as '{filename}'")
    
    plt.show()
    return fig

def main():
    """Main analysis function"""
    
    print("🌞 SEPTEMBER 6, 2017 X9.3 SOLAR FLARE - COMPLETE ANALYSIS")
    print("=" * 70)
    print("Largest flare of Solar Cycle 24 - Full EVEREST Model Analysis\\n")
    
    try:
        # Load data
        period_data, flare_time = load_september_6_data()
        
        # Prepare sequences
        sequences, timestamps = prepare_sequences(period_data)
        
        # Load model
        print("\\n🤖 Loading EVEREST model...")
        model = RETPlusWrapper(input_shape=(10, 9))
        model.load('tests/model_weights_EVEREST_72h_M5.pt')
        print("   ✅ Model loaded successfully")
        
        # Analyze model outputs
        results_df = analyze_model_outputs(model, sequences, timestamps, flare_time)
        
        # Comprehensive analysis
        analysis_results = analyze_prediction_quality(results_df, flare_time)
        
        # Create visualization
        fig = create_comprehensive_visualization(results_df, flare_time)
        
        print(f"\\n✅ COMPLETE ANALYSIS FINISHED!")
        print(f"🎯 September 6, 2017 X9.3 flare: {analysis_results['max_prob']:.2f}% max probability")
        print(f"📊 Comprehensive multi-modal analysis completed")
        print(f"📈 All missing components now addressed")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 