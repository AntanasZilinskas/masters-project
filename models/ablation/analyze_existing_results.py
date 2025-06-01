"""
Analyze Existing Ablation Results
"""

import json
import os
import pandas as pd

def analyze_existing_results():
    """Analyze existing ablation results."""
    
    results_dir = "ablation_results"
    
    print("🔬 EVEREST Ablation Results Analysis")
    print("=" * 50)
    
    # Load all JSON files
    results = {}
    file_mapping = {
        "full_model.json": "Full Model",
        "no_evidential_head.json": "No Evidential Head", 
        "no_evt_head.json": "No EVT Head",
        "no_precursor_head.json": "No Precursor Head",
        "mean_pooling.json": "Mean Pooling",
        "focal_only_loss.json": "Focal Only",
        "no_gamma_annealing.json": "No Gamma Annealing"
    }
    
    for filename, variant_name in file_mapping.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            results[variant_name] = data
            print(f"✅ {variant_name}: TSS={data['tss']:.4f}")
    
    # Baseline comparison
    if "Full Model" in results:
        baseline = results["Full Model"]
        print(f"\n📊 Component Effects (vs Full Model):")
        print("-" * 50)
        
        for variant, data in results.items():
            if variant != "Full Model":
                tss_change = baseline['tss'] - data['tss']
                brier_change = baseline['brier'] - data['brier']
                print(f"{variant:20}: TSS {tss_change:+.4f}, Brier {brier_change:+.4f}")
    
    return results

if __name__ == "__main__":
    results = analyze_existing_results() 