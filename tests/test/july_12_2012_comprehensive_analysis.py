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

# Load July 12, 2012 X1.4 flare data
def load_july_12_data():
    """Load data around July 12, 2012 X1.4 flare"""
    
    # Load the data file
    df = pd.read_csv('/Users/antanaszilinskas/Github/masters-project/Nature_data/training_data_M5_72.csv')
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['DATE__OBS'], utc=True)
    
    # Filter to July 12, 2012 flare period (72h before to 24h after)
    flare_time = pd.Timestamp('2012-07-12 16:52:00', tz='UTC')
    start_time = flare_time - timedelta(hours=72)
    end_time = flare_time + timedelta(hours=24)
    
    period_data = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()
    print(f"Loaded {len(period_data)} data points from {start_time} to {end_time}")
    
    return period_data, flare_time

def prepare_sequences(data, sequence_length=10):
    """Prepare sequences for the model with correct 9 features"""
    
    # The 9 features used by the model (from models/utils.py)
    feature_columns = [
        'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE',
        'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
    ]
    
    # Extract features and normalize
    features = data[feature_columns].values
    
    # Simple standardization
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    # Create sequences
    sequences = []
    timestamps = []
    
    for i in range(len(features) - sequence_length + 1):
        seq = features[i:i+sequence_length]
        sequences.append(seq)
        timestamps.append(data.iloc[i+sequence_length-1]['timestamp'])
    
    return np.array(sequences), timestamps

def analyze_model_outputs(model, sequences, timestamps, flare_time):
    """Comprehensive analysis of all model outputs"""
    
    model.model.eval()
    
    # Get device from the model
    device = next(model.model.parameters()).device
    print(f"Model is on device: {device}")
    
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
        # Higher xi means heavier tail (more extreme events likely)
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
    
    # Determine number of subplots based on available data
    n_plots = 2  # Always have probability and logits
    if 'total_uncertainty' in results_df.columns:
        n_plots += 1
    if 'tail_risk' in results_df.columns:
        n_plots += 1
    if 'precursor_score' in results_df.columns:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # 1. Main flare probability
    ax = axes[plot_idx]
    ax.plot(results_df['timestamp'], results_df['probability'], 'b-', linewidth=2, label='Flare Probability')
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X1.4 Flare')
    ax.set_ylabel('Flare Probability', fontsize=12)
    ax.set_title('EVEREST Model Outputs for July 12, 2012 X1.4 Flare', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)
    plot_idx += 1
    
    # 2. Logits (raw model output)
    ax = axes[plot_idx]
    ax.plot(results_df['timestamp'], results_df['logits'], 'g-', linewidth=2, label='Logits')
    ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X1.4 Flare')
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
        
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X1.4 Flare')
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
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X1.4 Flare')
        ax.set_ylabel('Tail Risk', fontsize=12)
        ax.set_title('Extreme Value Theory (EVT) Tail Risk', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plot_idx += 1
    
    # 5. Precursor score if available
    if 'precursor_score' in results_df.columns:
        ax = axes[plot_idx]
        ax.plot(results_df['timestamp'], results_df['precursor_score'], 'teal', linewidth=2, label='Precursor Score')
        ax.axvline(flare_time, color='red', linestyle='--', linewidth=2, label='X1.4 Flare')
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
    plt.savefig('july_12_2012_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive analysis plot as 'july_12_2012_comprehensive_analysis.png'")
    
    return fig

def analyze_prediction_quality(results_df, flare_time):
    """Analyze prediction quality and alert characteristics"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVEREST MODEL ANALYSIS")
    print("Event: July 12, 2012 X1.4 Flare")
    print("="*60)
    
    # Basic probability statistics
    print(f"\nPROBABILITY STATISTICS:")
    print(f"Mean probability: {results_df['probability'].mean():.4f}")
    print(f"Max probability: {results_df['probability'].max():.4f}")
    print(f"Min probability: {results_df['probability'].min():.4f}")
    print(f"Std probability: {results_df['probability'].std():.4f}")
    
    # Find when model first alerts at different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    print(f"\nALERT LEAD TIMES:")
    for threshold in thresholds:
        alerts = results_df[results_df['probability'] >= threshold]
        if len(alerts) > 0:
            first_alert = alerts.iloc[0]
            lead_time = first_alert['hours_to_flare']
            print(f"τ = {threshold:4.2f}: {lead_time:5.1f}h lead time ({first_alert['timestamp'].strftime('%m/%d %H:%M')})")
        else:
            print(f"τ = {threshold:4.2f}: No alerts")
    
    # Evidential analysis if available
    if 'total_uncertainty' in results_df.columns:
        print(f"\nEVIDENTIAL UNCERTAINTY ANALYSIS:")
        
        # Filter finite values
        mask = np.isfinite(results_df['epistemic_uncertainty']) & np.isfinite(results_df['aleatoric_uncertainty'])
        if mask.sum() > 0:
            print(f"Mean epistemic uncertainty: {results_df.loc[mask, 'epistemic_uncertainty'].mean():.4f}")
            print(f"Mean aleatoric uncertainty: {results_df.loc[mask, 'aleatoric_uncertainty'].mean():.4f}")
            
            # Correlation with probability
            corr_epist = np.corrcoef(results_df.loc[mask, 'probability'], results_df.loc[mask, 'epistemic_uncertainty'])[0,1]
            corr_aleat = np.corrcoef(results_df.loc[mask, 'probability'], results_df.loc[mask, 'aleatoric_uncertainty'])[0,1]
            print(f"Correlation: Prob vs Epistemic = {corr_epist:.3f}, Prob vs Aleatoric = {corr_aleat:.3f}")
    
    # EVT analysis if available
    if 'tail_risk' in results_df.columns:
        print(f"\nEXTREME VALUE THEORY ANALYSIS:")
        print(f"Mean tail risk: {results_df['tail_risk'].mean():.4f}")
        print(f"Max tail risk: {results_df['tail_risk'].max():.4f}")
        print(f"Mean shape parameter (ξ): {results_df['evt_xi'].mean():.4f}")  
        print(f"Mean scale parameter (σ): {results_df['evt_sigma'].mean():.4f}")
        
        # Correlation with probability
        corr_tail = np.corrcoef(results_df['probability'], results_df['tail_risk'])[0,1]
        print(f"Correlation: Prob vs Tail Risk = {corr_tail:.3f}")
    
    # Precursor analysis if available
    if 'precursor_score' in results_df.columns:
        print(f"\nPRECURSOR ANALYSIS:")
        print(f"Mean precursor score: {results_df['precursor_score'].mean():.4f}")
        print(f"Max precursor score: {results_df['precursor_score'].max():.4f}")
        
        # Correlation with main probability
        corr_prec = np.corrcoef(results_df['probability'], results_df['precursor_score'])[0,1]
        print(f"Correlation: Main Prob vs Precursor = {corr_prec:.3f}")
    
    # Pre-flare behavior analysis
    print(f"\nPRE-FLARE BEHAVIOR (Last 24 hours):")
    pre_flare = results_df[results_df['hours_to_flare'] <= 24]
    if len(pre_flare) > 0:
        print(f"Mean probability (24h before): {pre_flare['probability'].mean():.4f}")
        print(f"Max probability (24h before): {pre_flare['probability'].max():.4f}")
        print(f"Probability trend: {np.polyfit(range(len(pre_flare)), pre_flare['probability'], 1)[0]:.6f}/hour")
    
    return results_df

def main():
    # Load data
    print("Loading July 12, 2012 X1.4 flare data...")
    data, flare_time = load_july_12_data()
    
    # Prepare sequences
    print("Preparing sequences...")
    sequences, timestamps = prepare_sequences(data)
    print(f"Created {len(sequences)} sequences")
    
    # Load model
    print("Loading EVEREST model...")
    model = RETPlusWrapper(input_shape=(10, 9))
    model.load('/Users/antanaszilinskas/Github/masters-project/tests/model_weights_EVEREST_72h_M5.pt')
    
    # Analyze all model outputs
    print("Analyzing comprehensive model outputs...")
    results_df = analyze_model_outputs(model, sequences, timestamps, flare_time)
    
    # Create visualization
    print("Creating comprehensive visualization...")
    fig = create_comprehensive_visualization(results_df, flare_time)
    
    # Analyze prediction quality
    analyze_prediction_quality(results_df, flare_time)
    
    # Save detailed results
    results_df.to_csv('july_12_2012_comprehensive_results.csv', index=False)
    print(f"\nSaved detailed results to 'july_12_2012_comprehensive_results.csv'")
    
    plt.show()

if __name__ == "__main__":
    main() 