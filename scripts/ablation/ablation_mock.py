#!/usr/bin/env python
'''
Mock Ablation Study for SolarKnowledge Model
This script generates sample ablation study results without actual training.

Author: Antanas Zilinskas
'''

import os
import json
import numpy as np
from datetime import datetime

# Configurations matching the original ablation study
CONFIGURATIONS = [
    {
        'name': 'Full model',
        'description': '(Conv1D + BN) + LSTM + 4 TEBs + heavy dropout',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No LSTM',
        'description': 'only conv + BN, then TEBs',
        'use_conv': True,
        'use_lstm': False,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No conv',
        'description': 'BN then LSTM',
        'use_conv': False,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'Reduced TEBs',
        'description': '2 layers instead of 4',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 2,
        'dropout_rate': 0.2,
        'use_class_weighting': True
    },
    {
        'name': 'No class weighting',
        'description': 'No class weights for imbalanced data',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.2,
        'use_class_weighting': False
    },
    {
        'name': 'Light dropout',
        'description': 'dropout = 0.1 (lighter)',
        'use_conv': True,
        'use_lstm': True,
        'teb_layers': 4,
        'dropout_rate': 0.1,
        'use_class_weighting': True
    }
]

# TSS values from the table in your LaTeX document
MOCK_TSS_VALUES = {
    'Full model': 0.872,
    'No LSTM': 0.856,
    'No conv': 0.849,
    'Reduced TEBs': 0.838,
    'No class weighting': 0.785,
    'Light dropout': 0.810
}


def generate_mock_results(time_window="24", flare_class="M"):
    """Generate mock ablation study results"""
    results = []

    print(
        f"Generating mock results for {time_window}h {flare_class}-class prediction...")

    for config in CONFIGURATIONS:
        # Get TSS value from mock data
        tss = MOCK_TSS_VALUES[config['name']]

        # Create result entry
        result = {
            'config_name': config['name'],
            'description': config['description'],
            'tss': tss,
            'use_conv': config['use_conv'],
            'use_lstm': config['use_lstm'],
            'teb_layers': config['teb_layers'],
            'dropout_rate': config['dropout_rate'],
            'use_class_weighting': config['use_class_weighting'],
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        results.append(result)
        print(f"Configuration: {config['name']}")
        print(f"Mock TSS: {tss:.4f}")

    # Save results
    output_dir = "results/ablation"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"ablation_results_{time_window}h_{flare_class}_class.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nMock results saved to {output_file}")

    # Print formatted summary for LaTeX table
    print("\n=== ABLATION STUDY SUMMARY ===")
    print("Results for LaTeX table:")
    print(r"\begin{tabular}{l|c}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{TSS} \\")
    print(r"\midrule")

    for result in results:
        if result['config_name'] == 'Full model':
            print(
                f"Full model: {result['description']} & {result['tss']:.3f}\\\\")
        else:
            print(
                f"\\quad - {result['description']} & {result['tss']:.3f}\\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    return results


if __name__ == "__main__":
    print("Generating mock ablation study results...")
    results = generate_mock_results()

    print("\nFinal TSS values:")
    for result in results:
        print(f"{result['config_name']}: {result['tss']:.3f}")

    print("\nYou can now run the visualization script:")
    print("python visualize_ablation.py")
