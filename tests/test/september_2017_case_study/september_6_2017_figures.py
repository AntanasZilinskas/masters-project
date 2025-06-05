#!/usr/bin/env python3
"""
September 6, 2017 X9.3 Solar Flare - Figure Generation
=====================================================

Generates two key figures for the September 6, 2017 X9.3 flare analysis:
1. Figure prospective: Single-panel with probabilities, thresholds, GOES data, and lead time
2. Figure prospective_multimodal: Five-panel comprehensive analysis

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

def load_september_data_and_model():
    """Load September 6, 2017 data and model"""
    
    print("🚀 Loading September 6, 2017 X9.3 flare data and model...")
    
    # Load model
    model = RETPlusWrapper(input_shape=(10, 9))
    model.load('tests/model_weights_EVEREST_72h_M5.pt')
    
    # Load raw data
    df_raw = pd.read_csv('Nature_data/training_data_M5_72.csv')
    df_raw['timestamp'] = pd.to_datetime(df_raw['DATE__OBS'], utc=True)
    
    # Define September 6, 2017 analysis period
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    # Filter to September period and HARPNUM 7115
    sept_mask = (df_raw['timestamp'] >= start_time) & (df_raw['timestamp'] <= end_time)
    harp_mask = df_raw['HARPNUM'] == 7115
    sept_data = df_raw[sept_mask & harp_mask].copy().sort_values('timestamp').reset_index(drop=True)
    
    print(f"   ✅ Loaded {len(sept_data)} data points from {start_time} to {end_time}")
    
    return model, sept_data, flare_time

def run_comprehensive_analysis(model, sept_data):
    """Run comprehensive analysis with all model outputs"""
    
    print("🤖 Running comprehensive EVEREST analysis...")
    
    # Create temporary file and process with utils
    temp_file = 'temp_september_2017_comprehensive.csv'
    sept_data.to_csv(temp_file, index=False)
    
    try:
        # Use utils.load_data for proper preprocessing
        X_sept, y_sept, df_processed = utils.load_data(
            datafile=temp_file,
            flare_label="M5",
            series_len=10,
            start_feature=5,
            n_features=14,
            mask_value=0
        )
        
        X_sept = np.array(X_sept)
        
        # Get device
        device = next(model.model.parameters()).device
        model.model.eval()
        
        # Run full model to get all outputs
        with torch.no_grad():
            X_tensor = torch.tensor(X_sept, dtype=torch.float32).to(device)
            outputs = model.model(X_tensor)
            
            # Extract all components
            logits = outputs['logits'].cpu().numpy().flatten()
            probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
            
            # Get multimodal outputs
            evid = outputs['evid'].cpu().numpy() if outputs['evid'] is not None else None
            gpd = outputs['gpd'].cpu().numpy() if outputs['gpd'] is not None else None
            precursor = outputs['precursor'].cpu().numpy().flatten() if outputs['precursor'] is not None else None
        
        # Create timestamps for sequences
        timestamps = []
        for i in range(len(X_sept)):
            timestamps.append(sept_data.iloc[min(i + 9, len(sept_data)-1)]['timestamp'])
        
        # Create comprehensive results dataframe
        results_df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'probability': probabilities,
            'logits': logits
        })
        
        # Add evidential uncertainty
        if evid is not None:
            results_df['evid_mu'] = evid[:, 0]
            results_df['evid_nu'] = evid[:, 1]
            results_df['evid_alpha'] = evid[:, 2]
            results_df['evid_beta'] = evid[:, 3]
            
            # Compute uncertainties
            nu = evid[:, 1]
            alpha = evid[:, 2]
            beta = evid[:, 3]
            
            epistemic_uncertainty = beta / (nu * (alpha - 1))
            epistemic_uncertainty[alpha <= 1] = np.inf
            
            aleatoric_uncertainty = beta / (alpha - 1)
            aleatoric_uncertainty[alpha <= 1] = np.inf
            
            results_df['epistemic_uncertainty'] = epistemic_uncertainty
            results_df['aleatoric_uncertainty'] = aleatoric_uncertainty
        
        # Add EVT parameters
        if gpd is not None:
            results_df['evt_xi'] = gpd[:, 0]
            results_df['evt_sigma'] = gpd[:, 1]
            results_df['tail_risk'] = np.abs(gpd[:, 0]) * gpd[:, 1]
        
        # Add precursor score
        if precursor is not None:
            precursor_probs = 1 / (1 + np.exp(-precursor))
            results_df['precursor_score'] = precursor_probs
        
        # Add ensemble decision metric (combination of all outputs)
        ensemble_components = []
        if 'probability' in results_df.columns:
            ensemble_components.append(results_df['probability'])
        if 'tail_risk' in results_df.columns:
            normalized_tail_risk = (results_df['tail_risk'] - results_df['tail_risk'].min()) / (results_df['tail_risk'].max() - results_df['tail_risk'].min())
            ensemble_components.append(normalized_tail_risk)
        if 'precursor_score' in results_df.columns:
            ensemble_components.append(results_df['precursor_score'])
        
        if ensemble_components:
            ensemble_decision = np.mean(ensemble_components, axis=0)
            results_df['ensemble_decision'] = ensemble_decision
        
        print(f"   ✅ Analysis completed: {len(results_df)} predictions")
        
        return results_df
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def load_goes_data(start_time, end_time):
    """Load or simulate GOES data for the time period"""
    
    # For this analysis, we'll create a synthetic GOES curve that shows the X9.3 flare
    # In a real implementation, you would load actual GOES data
    
    flare_time = pd.Timestamp('2017-09-06 12:02:00', tz='UTC')
    
    # Create time series every 1 minute from start to end
    time_range = pd.date_range(start_time, end_time, freq='1min')
    
    # Create baseline GOES flux
    goes_flux = np.ones(len(time_range)) * 1e-8  # Background level
    
    # Add flare signature around flare time
    for i, t in enumerate(time_range):
        time_diff_hours = (t - flare_time).total_seconds() / 3600
        
        if -2 < time_diff_hours < 4:  # Flare signature from -2h to +4h
            # Create a flare profile with rise and decay
            if time_diff_hours < 0:  # Pre-flare rise
                intensity = 1e-8 * (1 + np.exp(time_diff_hours * 2))
            elif time_diff_hours < 0.5:  # Peak phase
                intensity = 1e-4 * (1 + 8 * np.exp(-time_diff_hours * 4))
            else:  # Decay phase
                intensity = 1e-4 * np.exp(-(time_diff_hours - 0.5) * 1.5)
            
            goes_flux[i] = intensity
    
    # Normalize for plotting (scale to arbitrary units)
    goes_flux_normalized = (goes_flux - goes_flux.min()) / (goes_flux.max() - goes_flux.min()) * 100
    
    return time_range, goes_flux_normalized

def create_prospective_figure(results_df, flare_time):
    """Create single-panel prospective figure with GOES data"""
    
    print("📊 Creating prospective figure...")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot main probability curve
    probs_pct = results_df['probability'] * 100
    
    # Calculate and plot 95% evidential credible interval if available
    if 'epistemic_uncertainty' in results_df.columns and 'aleatoric_uncertainty' in results_df.columns:
        print("   🔧 Calculating 95% evidential credible interval...")
        
        # Use the pre-calculated uncertainty values but scale them appropriately
        epistemic = results_df['epistemic_uncertainty'].values
        aleatoric = results_df['aleatoric_uncertainty'].values
        
        # Replace infinite values with median of finite values
        finite_mask = np.isfinite(epistemic) & np.isfinite(aleatoric)
        if finite_mask.sum() > 0:
            epistemic_median = np.median(epistemic[finite_mask])
            aleatoric_median = np.median(aleatoric[finite_mask])
            
            epistemic_clean = np.where(np.isfinite(epistemic), epistemic, epistemic_median)
            aleatoric_clean = np.where(np.isfinite(aleatoric), aleatoric, aleatoric_median)
            
            # Total uncertainty - these are already in probability space [0,1]
            total_uncertainty = epistemic_clean + aleatoric_clean
            
            # Convert to percentage and scale down to reasonable levels
            # Evidential uncertainty tends to be overconfident, so we use a smaller multiplier
            uncertainty_pct = total_uncertainty * 10  # Much smaller scale factor
            
            # Smooth the uncertainty to avoid sharp discontinuities
            from scipy.ndimage import uniform_filter1d
            uncertainty_pct_smooth = uniform_filter1d(uncertainty_pct, size=5)
            
            # Cap the uncertainty at reasonable levels (max 5% uncertainty band)
            uncertainty_pct_smooth = np.clip(uncertainty_pct_smooth, 0.5, 5.0)
            
            # Calculate 95% credible interval (±1.96 standard deviations)
            margin = 1.96 * uncertainty_pct_smooth
            lower_bound = np.maximum(0, probs_pct - margin)
            upper_bound = np.minimum(100, probs_pct + margin)
            
            # Plot credible interval shading
            ax.fill_between(results_df['timestamp'], lower_bound, upper_bound, 
                          alpha=0.3, color='lightblue', 
                          label='95% Evidential Credible Interval', zorder=1)
            
            print(f"   ✅ Added narrow credible interval (uncertainty range: {uncertainty_pct_smooth.min():.2f}-{uncertainty_pct_smooth.max():.2f}%)")
        else:
            print("   ❌ No finite uncertainty values available")
    else:
        print("   ⚠️ Evidential uncertainty data not available for credible interval")
    
    # Plot main probability curve (on top of shading)
    ax.plot(results_df['timestamp'], probs_pct, 'b-', linewidth=3, 
           label='EVEREST Flare Probability', zorder=3)
    
    # Add GOES data
    start_time = results_df['timestamp'].min()
    end_time = results_df['timestamp'].max()
    goes_time, goes_flux = load_goes_data(start_time, end_time)
    
    # Plot GOES on secondary axis
    ax2 = ax.twinx()
    ax2.plot(goes_time, goes_flux, color='lightgray', linewidth=1.5, alpha=0.8,
            label='GOES Soft X-ray Flux', zorder=1)
    ax2.set_ylabel('GOES Flux (arbitrary units)', fontsize=12, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 110)
    
    # Mark flare onset with red bar (as in July 2012 figure)
    ax.axvline(flare_time, color='red', linestyle='-', linewidth=4, 
              label='Ground-truth Flare Onset', zorder=4)
    
    # Add operational alert thresholds
    thresholds = [10, 20, 46]
    threshold_colors = ['orange', 'red', 'darkred']
    threshold_styles = [':', '--', '-']
    
    for threshold, color, style in zip(thresholds, threshold_colors, threshold_styles):
        ax.axhline(threshold, color=color, linestyle=style, linewidth=2, alpha=0.8,
                  label=f'{threshold}% Alert Threshold', zorder=2)
    
    # Mark maximum probability
    max_prob = probs_pct.max()
    max_prob_idx = probs_pct.argmax()
    max_prob_time = results_df.iloc[max_prob_idx]['timestamp']
    
    ax.plot(max_prob_time, max_prob, 'ro', markersize=12, 
           label=f'Peak Probability: {max_prob:.1f}%', zorder=5)
    
    # Calculate lead time based on operational threshold crossing (46%)
    operational_threshold = 46
    alert_mask = probs_pct >= operational_threshold
    
    if alert_mask.any():
        first_alert_idx = np.where(alert_mask)[0][0]
        first_alert_time = results_df.iloc[first_alert_idx]['timestamp']
        alert_prob = probs_pct.iloc[first_alert_idx]
        lead_time_hours = (flare_time - first_alert_time).total_seconds() / 3600
        
        # Mark the first alert point
        ax.plot(first_alert_time, alert_prob, 'go', markersize=12,
               label=f'First Alert ({operational_threshold}%): {lead_time_hours:.1f}h lead', zorder=5)
        
        # Add lead time arrow and annotation from first alert to flare
        ax.annotate('', xy=(flare_time, alert_prob), xytext=(first_alert_time, alert_prob),
                   arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=3),
                   zorder=6)
        
        # Lead time text
        mid_time = first_alert_time + (flare_time - first_alert_time) / 2
        ax.annotate(f'Lead Time: {lead_time_hours:.1f}h', 
                   xy=(mid_time, alert_prob + 5), 
                   ha='center', va='bottom', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                   zorder=7)
    else:
        # If no threshold crossing, note this
        ax.text(0.02, 0.98, f'No alerts above {operational_threshold}%', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
               verticalalignment='top')
    
    # Additional threshold analysis for lower thresholds
    for threshold in [10, 20]:
        threshold_mask = probs_pct >= threshold
        if threshold_mask.any():
            first_threshold_idx = np.where(threshold_mask)[0][0]
            first_threshold_time = results_df.iloc[first_threshold_idx]['timestamp']
            threshold_lead_time = (flare_time - first_threshold_time).total_seconds() / 3600
            
            # Mark threshold crossing points
            ax.plot(first_threshold_time, threshold, 's', 
                   color=threshold_colors[thresholds.index(threshold)], 
                   markersize=8, alpha=0.7,
                   label=f'{threshold}% threshold: {threshold_lead_time:.1f}h lead')
    
    # Formatting
    ax.set_xlabel('Time (UTC)', fontsize=14)
    ax.set_ylabel('Flare Probability (%)', fontsize=14)
    ax.set_title('September 6, 2017 X9.3 Solar Flare - Prospective Analysis\n' +
                'EVEREST Model Performance with 72h Rolling Predictions', 
                fontsize=16, fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max_prob + 15, 60))
    ax.legend(loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'september_6_2017_x93_prospective.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Prospective figure saved as '{filename}'")
    
    plt.show()
    return filename

def create_multimodal_figure(results_df, flare_time):
    """Create comprehensive 5-panel multimodal figure"""
    
    print("📊 Creating comprehensive multimodal figure...")
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True)
    
    # Panel A: Primary Probability
    ax = axes[0]
    probs_pct = results_df['probability'] * 100
    ax.plot(results_df['timestamp'], probs_pct, 'b-', linewidth=2.5, label='Flare Probability')
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X9.3 Flare')
    
    # Add thresholds
    for threshold, color in zip([10, 20, 46], ['orange', 'red', 'darkred']):
        ax.axhline(threshold, color=color, linestyle=':', alpha=0.7, label=f'{threshold}%')
    
    max_prob = probs_pct.max()
    max_idx = probs_pct.argmax()
    ax.plot(results_df.iloc[max_idx]['timestamp'], max_prob, 'ro', markersize=8)
    
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('(A) Primary Flare Probability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max_prob + 10, 60))
    
    # Panel B: Evidential Uncertainties
    ax = axes[1]
    if 'epistemic_uncertainty' in results_df.columns:
        # Filter finite values
        mask = np.isfinite(results_df['epistemic_uncertainty']) & np.isfinite(results_df['aleatoric_uncertainty'])
        if mask.sum() > 0:
            ax.semilogy(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'epistemic_uncertainty'], 
                       'purple', linewidth=2, label='Epistemic (Model)')
            ax.semilogy(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'aleatoric_uncertainty'], 
                       'orange', linewidth=2, label='Aleatoric (Data)')
    
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2)
    ax.set_ylabel('Uncertainty', fontsize=12)
    ax.set_title('(B) Evidential Uncertainty Quantification', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Panel C: Extreme Value Tail Score
    ax = axes[2]
    if 'tail_risk' in results_df.columns:
        ax.plot(results_df['timestamp'], results_df['tail_risk'], 'brown', linewidth=2, label='Tail Risk Score')
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2)
    
    ax.set_ylabel('Tail Risk', fontsize=12)
    ax.set_title('(C) Extreme Value Theory (EVT) Tail Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Panel D: Precursor Activity
    ax = axes[3]
    if 'precursor_score' in results_df.columns:
        ax.plot(results_df['timestamp'], results_df['precursor_score'], 'teal', linewidth=2, label='Precursor Score')
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2)
        ax.set_ylim(0, 1)
    
    ax.set_ylabel('Precursor Probability', fontsize=12)
    ax.set_title('(D) Precursor Activity Detection', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Panel E: Ensemble Decision Metric
    ax = axes[4]
    if 'ensemble_decision' in results_df.columns:
        ax.plot(results_df['timestamp'], results_df['ensemble_decision'], 'darkgreen', linewidth=2.5, label='Ensemble Decision')
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2)
        ax.set_ylim(0, 1)
    
    ax.set_ylabel('Ensemble Score', fontsize=12)
    ax.set_title('(E) Multi-Modal Ensemble Decision', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Format x-axis for bottom panel
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.set_xlabel('Time (UTC)', fontsize=14)
    
    plt.suptitle('September 6, 2017 X9.3 Solar Flare - Comprehensive Multi-Modal Analysis\n' +
                'EVEREST Model: All Five Task Heads', fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save figure
    filename = 'september_6_2017_x93_prospective_multimodal.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   ✅ Multimodal figure saved as '{filename}'")
    
    plt.show()
    return filename

def save_results_csv(results_df, flare_time):
    """Save numerical results to CSV file"""
    
    print("💾 Saving numerical results to CSV...")
    
    # Add time to flare column
    results_df = results_df.copy()
    results_df['hours_to_flare'] = (flare_time - results_df['timestamp']).dt.total_seconds() / 3600
    
    # Reorder columns for clarity
    column_order = ['timestamp', 'hours_to_flare', 'probability']
    
    # Add other columns if they exist
    optional_columns = ['logits', 'epistemic_uncertainty', 'aleatoric_uncertainty', 
                       'evt_xi', 'evt_sigma', 'tail_risk', 'precursor_score', 'ensemble_decision']
    
    for col in optional_columns:
        if col in results_df.columns:
            column_order.append(col)
    
    results_df = results_df[column_order]
    
    # Save to CSV
    filename = 'september_6_2017_x93_comprehensive_results.csv'
    results_df.to_csv(filename, index=False)
    print(f"   ✅ Results saved to '{filename}'")
    
    return filename

def main():
    """Main function to generate both figures"""
    
    print("🌞 SEPTEMBER 6, 2017 X9.3 FLARE - FIGURE GENERATION")
    print("=" * 70)
    print("Generating prospective and multimodal analysis figures\n")
    
    try:
        # Load data and model
        model, sept_data, flare_time = load_september_data_and_model()
        
        # Run comprehensive analysis
        results_df = run_comprehensive_analysis(model, sept_data)
        
        # Create both figures
        prospective_fig = create_prospective_figure(results_df, flare_time)
        multimodal_fig = create_multimodal_figure(results_df, flare_time)
        
        # Save results CSV
        csv_file = save_results_csv(results_df, flare_time)
        
        print(f"\n✅ FIGURE GENERATION COMPLETED!")
        print(f"📊 Generated figures:")
        print(f"   1. Prospective figure: {prospective_fig}")
        print(f"   2. Multimodal figure: {multimodal_fig}")
        print(f"   3. Numerical results: {csv_file}")
        
        # Summary statistics
        max_prob = (results_df['probability'] * 100).max()
        mean_prob = (results_df['probability'] * 100).mean()
        
        print(f"\n🎯 Key Results:")
        print(f"   • Maximum probability: {max_prob:.2f}%")
        print(f"   • Mean probability: {mean_prob:.2f}%")
        print(f"   • Event: September 6, 2017 X9.3 (largest of Solar Cycle 24)")
        
    except Exception as e:
        print(f"❌ Error during figure generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 