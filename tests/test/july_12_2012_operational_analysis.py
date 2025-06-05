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
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def load_july_12_data():
    """Load data around July 12, 2012 X1.4 flare"""
    df = pd.read_csv('/Users/antanaszilinskas/Github/masters-project/Nature_data/training_data_M5_72.csv')
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
    feature_columns = [
        'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTBSQ', 'R_VALUE',
        'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH'
    ]
    
    features = data[feature_columns].values
    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
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
    device = next(model.model.parameters()).device
    
    with torch.no_grad():
        X_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
        outputs = model.model(X_tensor)
        
        # Extract all components
        logits = outputs['logits'].cpu().numpy().flatten()
        probabilities = 1 / (1 + np.exp(-logits))
        
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
        
        nu = evid[:, 1]
        alpha = evid[:, 2] 
        beta = evid[:, 3]
        
        epistemic_uncertainty = beta / (nu * (alpha - 1))
        epistemic_uncertainty[alpha <= 1] = np.inf
        
        aleatoric_uncertainty = beta / (alpha - 1)
        aleatoric_uncertainty[alpha <= 1] = np.inf
        
        results_df['epistemic_uncertainty'] = epistemic_uncertainty
        results_df['aleatoric_uncertainty'] = aleatoric_uncertainty
        results_df['total_uncertainty'] = epistemic_uncertainty + aleatoric_uncertainty
    
    # Add EVT parameters if available
    if gpd is not None:
        results_df['evt_xi'] = gpd[:, 0]  # Shape parameter
        results_df['evt_sigma'] = gpd[:, 1]  # Scale parameter
        results_df['tail_risk'] = np.abs(gpd[:, 0]) * gpd[:, 1]
    
    # Add precursor score if available
    if precursor is not None:
        precursor_probs = 1 / (1 + np.exp(-precursor))
        results_df['precursor_score'] = precursor_probs
    
    # Calculate time to flare
    results_df['hours_to_flare'] = (flare_time - results_df['timestamp']).dt.total_seconds() / 3600
    
    return results_df

def ensemble_decision_making(results_df, flare_time):
    """Analyze how different model outputs combine for final decisions"""
    
    print("\n" + "="*80)
    print("OPERATIONAL DECISION-MAKING ANALYSIS")
    print("How EVEREST combines multiple outputs for final flare prediction")
    print("="*80)
    
    # 1. THRESHOLD ANALYSIS
    print("\n🎯 THRESHOLD SELECTION ANALYSIS:")
    print("Finding optimal operating points for space weather forecasting")
    
    # Create binary labels (1 for 24h before flare, 0 otherwise)
    results_df['true_label'] = (results_df['hours_to_flare'] <= 24) & (results_df['hours_to_flare'] >= 0)
    
    # Calculate various performance metrics at different thresholds
    thresholds = np.arange(0.01, 0.20, 0.01)
    threshold_analysis = []
    
    for thresh in thresholds:
        predictions = (results_df['probability'] >= thresh).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (results_df['true_label'] == 1))
        fp = np.sum((predictions == 1) & (results_df['true_label'] == 0))
        tn = np.sum((predictions == 0) & (results_df['true_label'] == 0))
        fn = np.sum((predictions == 0) & (results_df['true_label'] == 1))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        tss = recall + specificity - 1  # True Skill Statistic
        
        # Find lead time if any alerts
        alerts = results_df[results_df['probability'] >= thresh]
        lead_time = alerts.iloc[0]['hours_to_flare'] if len(alerts) > 0 else np.nan
        
        threshold_analysis.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'f1': f1,
            'tss': tss,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'lead_time': lead_time
        })
    
    threshold_df = pd.DataFrame(threshold_analysis)
    
    # Find optimal thresholds
    best_f1_idx = threshold_df['f1'].idxmax()
    best_tss_idx = threshold_df['tss'].idxmax()
    
    print(f"\n📊 OPTIMAL OPERATING POINTS:")
    print(f"Best F1 Score: τ = {threshold_df.iloc[best_f1_idx]['threshold']:.3f}")
    print(f"  • F1: {threshold_df.iloc[best_f1_idx]['f1']:.3f}")
    print(f"  • Precision: {threshold_df.iloc[best_f1_idx]['precision']:.3f}")
    print(f"  • Recall: {threshold_df.iloc[best_f1_idx]['recall']:.3f}")
    print(f"  • Lead Time: {threshold_df.iloc[best_f1_idx]['lead_time']:.1f}h")
    
    print(f"\nBest TSS: τ = {threshold_df.iloc[best_tss_idx]['threshold']:.3f}")
    print(f"  • TSS: {threshold_df.iloc[best_tss_idx]['tss']:.3f}")
    print(f"  • Precision: {threshold_df.iloc[best_tss_idx]['precision']:.3f}")
    print(f"  • Recall: {threshold_df.iloc[best_tss_idx]['recall']:.3f}")
    print(f"  • Lead Time: {threshold_df.iloc[best_tss_idx]['lead_time']:.1f}h")
    
    # 2. ENSEMBLE COMBINATION ANALYSIS
    print(f"\n🔗 ENSEMBLE COMBINATION STRATEGIES:")
    
    # Strategy 1: Weighted combination
    weights = {
        'main_prob': 0.7,      # Primary flare probability
        'uncertainty': 0.1,    # Lower uncertainty = higher confidence
        'tail_risk': 0.1,      # EVT tail risk
        'precursor': 0.1       # Precursor activity
    }
    
    # Normalize uncertainty (inverse - lower uncertainty = higher confidence)
    if 'total_uncertainty' in results_df.columns:
        finite_unc = results_df['total_uncertainty'][np.isfinite(results_df['total_uncertainty'])]
        if len(finite_unc) > 0:
            results_df['uncertainty_score'] = 1 / (1 + results_df['total_uncertainty'])
            results_df['uncertainty_score'] = results_df['uncertainty_score'].fillna(0)
        else:
            results_df['uncertainty_score'] = 0.5
    else:
        results_df['uncertainty_score'] = 0.5
    
    # Normalize tail risk
    if 'tail_risk' in results_df.columns:
        tail_min, tail_max = results_df['tail_risk'].min(), results_df['tail_risk'].max()
        results_df['tail_risk_norm'] = (results_df['tail_risk'] - tail_min) / (tail_max - tail_min + 1e-8)
    else:
        results_df['tail_risk_norm'] = 0.0
    
    # Normalize precursor (already 0-1)
    if 'precursor_score' in results_df.columns:
        prec_min, prec_max = results_df['precursor_score'].min(), results_df['precursor_score'].max()
        results_df['precursor_norm'] = (results_df['precursor_score'] - prec_min) / (prec_max - prec_min + 1e-8)
    else:
        results_df['precursor_norm'] = 0.0
    
    # Calculate ensemble score
    results_df['ensemble_score'] = (
        weights['main_prob'] * results_df['probability'] +
        weights['uncertainty'] * results_df['uncertainty_score'] +
        weights['tail_risk'] * results_df['tail_risk_norm'] +
        weights['precursor'] * results_df['precursor_norm']
    )
    
    print(f"Weighted Ensemble (weights: {weights}):")
    print(f"  • Max ensemble score: {results_df['ensemble_score'].max():.4f}")
    print(f"  • Mean ensemble score: {results_df['ensemble_score'].mean():.4f}")
    
    # Find when ensemble would alert
    ensemble_thresh = 0.1
    ensemble_alerts = results_df[results_df['ensemble_score'] >= ensemble_thresh]
    if len(ensemble_alerts) > 0:
        ensemble_lead = ensemble_alerts.iloc[0]['hours_to_flare']
        print(f"  • Ensemble alert (τ=0.1): {ensemble_lead:.1f}h lead time")
    
    # 3. DECISION RULES
    print(f"\n⚡ OPERATIONAL DECISION RULES:")
    print("Recommended thresholds for different use cases:")
    
    print(f"\n🚨 CONSERVATIVE (Minimize False Positives):")
    print(f"  • Main probability ≥ 0.10 (36h lead time)")
    print(f"  • Low epistemic uncertainty")
    print(f"  • For critical infrastructure protection")
    
    print(f"\n⚠️  BALANCED (Operational Forecasting):")
    print(f"  • Main probability ≥ 0.05 (37h lead time)")
    print(f"  • Ensemble score ≥ 0.08")
    print(f"  • For routine space weather operations")
    
    print(f"\n📢 SENSITIVE (Early Warning):")
    print(f"  • Main probability ≥ 0.02 (65h lead time)")
    print(f"  • Rising precursor activity")
    print(f"  • For research and preparation")
    
    return results_df, threshold_df

def create_operational_visualization(results_df, flare_time, threshold_df):
    """Create visualization focused on operational decision-making"""
    
    fig = plt.figure(figsize=(18, 20))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Main probability with decision thresholds
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results_df['timestamp'], results_df['probability'], 'b-', linewidth=3, label='Flare Probability')
    ax1.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='Conservative (τ=0.10)')
    ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Balanced (τ=0.05)')
    ax1.axhline(y=0.02, color='yellow', linestyle='--', alpha=0.7, label='Sensitive (τ=0.02)')
    ax1.axvline(flare_time, color='red', linestyle='-', linewidth=3, label='X1.4 Flare')
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title('EVEREST Operational Thresholds for July 12, 2012 X1.4 Flare', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 0.18)
    
    # 2. Uncertainty analysis
    ax2 = fig.add_subplot(gs[1, 0])
    if 'epistemic_uncertainty' in results_df.columns:
        mask = np.isfinite(results_df['epistemic_uncertainty'])
        if mask.sum() > 0:
            ax2.plot(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'epistemic_uncertainty'], 
                    'purple', linewidth=2, label='Epistemic')
            ax2.plot(results_df.loc[mask, 'timestamp'], results_df.loc[mask, 'aleatoric_uncertainty'], 
                    'orange', linewidth=2, label='Aleatoric')
    ax2.axvline(flare_time, color='red', linestyle='-', linewidth=2)
    ax2.set_ylabel('Uncertainty', fontsize=10)
    ax2.set_title('Model Confidence Assessment', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. Precursor analysis (with proper scaling)
    ax3 = fig.add_subplot(gs[1, 1])
    if 'precursor_score' in results_df.columns:
        # Scale precursor to percentage for visibility
        precursor_pct = results_df['precursor_score'] * 100
        ax3.plot(results_df['timestamp'], precursor_pct, 'teal', linewidth=2, label='Precursor Activity')
        ax3.axvline(flare_time, color='red', linestyle='-', linewidth=2)
        ax3.set_ylabel('Precursor Score (%)', fontsize=10)
        ax3.set_title('Precursor Activity Detection', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. EVT tail risk
    ax4 = fig.add_subplot(gs[2, 0])
    if 'tail_risk' in results_df.columns:
        ax4.plot(results_df['timestamp'], results_df['tail_risk'], 'brown', linewidth=2, label='Tail Risk')
        ax4.axvline(flare_time, color='red', linestyle='-', linewidth=2)
        ax4.set_ylabel('EVT Tail Risk', fontsize=10)
        ax4.set_title('Extreme Value Assessment', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # 5. Ensemble score
    ax5 = fig.add_subplot(gs[2, 1])
    if 'ensemble_score' in results_df.columns:
        ax5.plot(results_df['timestamp'], results_df['ensemble_score'], 'darkgreen', linewidth=2, label='Ensemble Score')
        ax5.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Alert Threshold')
        ax5.axvline(flare_time, color='red', linestyle='-', linewidth=2)
        ax5.set_ylabel('Ensemble Score', fontsize=10)
        ax5.set_title('Combined Decision Score', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # 6. Threshold analysis
    ax6 = fig.add_subplot(gs[3, :])
    ax6_twin = ax6.twinx()
    
    # Plot metrics
    ax6.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', linewidth=2, label='Precision')
    ax6.plot(threshold_df['threshold'], threshold_df['recall'], 'g-', linewidth=2, label='Recall')
    ax6.plot(threshold_df['threshold'], threshold_df['f1'], 'r-', linewidth=2, label='F1 Score')
    ax6.plot(threshold_df['threshold'], threshold_df['tss'], 'm-', linewidth=2, label='TSS')
    
    # Plot lead times on twin axis
    valid_lead = threshold_df.dropna(subset=['lead_time'])
    ax6_twin.plot(valid_lead['threshold'], valid_lead['lead_time'], 'k--', linewidth=2, label='Lead Time')
    
    ax6.set_xlabel('Threshold', fontsize=12)
    ax6.set_ylabel('Performance Metrics', fontsize=12)
    ax6_twin.set_ylabel('Lead Time (hours)', fontsize=12)
    ax6.set_title('Threshold vs Performance Trade-offs', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='center left')
    ax6_twin.legend(loc='center right')
    
    # 7. Decision timeline
    ax7 = fig.add_subplot(gs[4, :])
    
    # Mark different alert levels
    for thresh, color, label in [(0.02, 'yellow', 'Sensitive'), (0.05, 'orange', 'Balanced'), (0.10, 'red', 'Conservative')]:
        alerts = results_df[results_df['probability'] >= thresh]
        if len(alerts) > 0:
            first_alert = alerts.iloc[0]['timestamp']
            lead_time = alerts.iloc[0]['hours_to_flare']
            ax7.axvline(first_alert, color=color, linestyle='--', linewidth=2, alpha=0.8)
            ax7.text(first_alert, 0.8, f'{label}\n{lead_time:.1f}h', 
                    rotation=90, ha='right', va='top', fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax7.axvline(flare_time, color='red', linestyle='-', linewidth=3, label='X1.4 Flare')
    ax7.plot(results_df['timestamp'], results_df['probability'], 'b-', linewidth=2, alpha=0.7)
    ax7.set_ylabel('Probability', fontsize=12)
    ax7.set_title('Operational Alert Timeline', fontsize=12)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0, 0.18)
    
    # Format x-axis for all plots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax7]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax7.set_xlabel('Date (UTC)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('july_12_2012_operational_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved operational analysis plot as 'july_12_2012_operational_analysis.png'")
    
    return fig

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
    print("Analyzing model outputs...")
    results_df = analyze_model_outputs(model, sequences, timestamps, flare_time)
    
    # Operational decision analysis
    results_df, threshold_df = ensemble_decision_making(results_df, flare_time)
    
    # Create visualization
    print("Creating operational visualization...")
    fig = create_operational_visualization(results_df, flare_time, threshold_df)
    
    # Save results
    results_df.to_csv('july_12_2012_operational_results.csv', index=False)
    threshold_df.to_csv('july_12_2012_threshold_analysis.csv', index=False)
    print(f"\nSaved operational results to CSV files")
    
    plt.show()

if __name__ == "__main__":
    main() 