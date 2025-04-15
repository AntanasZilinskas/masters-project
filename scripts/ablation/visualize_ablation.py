#!/usr/bin/env python
'''
Visualize Ablation Study Results for SolarKnowledge Model
This script creates visualizations and formatted output from ablation study results.

Author: Antanas Zilinskas
'''

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn as sns

def load_ablation_results(filepath):
    """Load ablation study results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_bar_chart(results, metric='tss', output_dir="results/ablation"):
    """Create a bar chart showing ablation results"""
    # Extract configuration names and metric values
    config_names = []
    metric_values = []
    
    # Sort results so Full model is first, then others
    full_model = None
    other_models = []
    
    for result in results:
        if result['config_name'] == 'Full model':
            full_model = result
        else:
            other_models.append(result)
    
    # Sort other models by metric value (descending)
    other_models.sort(key=lambda x: x[metric], reverse=True)
    
    # Combine full model and sorted other models
    sorted_results = [full_model] + other_models if full_model else other_models
    
    for result in sorted_results:
        if result['config_name'] == 'Full model':
            config_names.append('Full model')
        else:
            # For ablated models, use a shorter description
            config_names.append(result['description'])
        metric_values.append(result[metric])
    
    # Create figure with larger size
    plt.figure(figsize=(12, 6))
    
    # Create bars with different colors for full model vs ablated versions
    colors = ['#1f77b4'] + ['#ff7f0e'] * (len(config_names) - 1)
    
    # Create bar chart
    bars = plt.bar(range(len(config_names)), metric_values, color=colors)
    
    # Add values on top of bars
    for i, v in enumerate(metric_values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    # Labels and title
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f'Ablation Study Results - Impact on {metric.upper()}', fontsize=14)
    
    # Set x-tick labels with rotation for readability
    plt.xticks(range(len(config_names)), config_names, rotation=45, ha='right', fontsize=10)
    
    # Adjust y-axis to start from a reasonable value for better visualization
    y_min = min(metric_values) * 0.9
    y_max = max(metric_values) * 1.05
    plt.ylim(y_min, y_max)
    
    # Add a horizontal line for the full model performance for reference
    if full_model:
        plt.axhline(y=full_model[metric], color='#1f77b4', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"ablation_{metric}_chart.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {output_file}")
    
    return output_file

def create_comparison_table(results, metric='tss'):
    """Create a formatted table comparing ablation results"""
    # Create DataFrame for better formatting
    df = pd.DataFrame(results)
    
    # Sort with full model first
    df = df.sort_values(by=['config_name'], key=lambda x: x.map({'Full model': 0}).fillna(1))
    
    # Format the table
    pd.set_option('display.max_colwidth', None)
    
    # Select and rename columns for the table
    table_df = df[['config_name', 'description', metric]].copy()
    table_df.columns = ['Configuration', 'Description', metric.upper()]
    
    # Format metric values
    table_df[metric.upper()] = table_df[metric.upper()].map('{:.3f}'.format)
    
    # Print table
    print("\nAblation Study Results:")
    print(table_df.to_string(index=False))
    
    # Create LaTeX table code
    print("\nLaTeX Table:")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Ablation of model components on the 24h $\\geq$M-class task. Removing or modifying certain parts significantly impacts TSS.}")
    print("\\label{tab:ablation}")
    print("\\small")
    print("\\begin{tabular}{l|c}")
    print("\\toprule")
    print("\\textbf{Configuration} & \\textbf{TSS} \\\\")
    print("\\midrule")
    
    # Print each row in LaTeX format
    for i, row in table_df.iterrows():
        if row['Configuration'] == 'Full model':
            print(f"Full model: {row['Description']} & {row[metric.upper()]}\\\\")
        else:
            print(f"\\quad - {row['Description']} & {row[metric.upper()]}\\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    return table_df

def create_heatmap(results, output_dir="results/ablation"):
    """Create a heatmap showing the impact of different ablations"""
    # Extract features and their impact
    features = {
        'LSTM': [r['use_lstm'] for r in results],
        'Conv1D': [r['use_conv'] for r in results],
        'TEBs=4': [r['teb_layers'] == 4 for r in results],
        'Class weighting': [r['use_class_weighting'] for r in results],
        'Heavy dropout': [r['dropout_rate'] >= 0.2 for r in results]
    }
    
    tss_values = [r['tss'] for r in results]
    
    # Create a matrix for heatmap
    matrix = []
    feature_names = list(features.keys())
    
    for i, result in enumerate(results):
        row = []
        for feature in feature_names:
            row.append(1 if features[feature][i] else 0)
        matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(matrix, columns=feature_names)
    
    # Add TSS as a column
    df['TSS'] = tss_values
    
    # Sort by TSS
    df = df.sort_values(by='TSS', ascending=False)
    
    # Create figure
    plt.figure(figsize=(10, len(results) * 0.8))
    
    # Create heatmap
    sns.heatmap(df[feature_names], annot=False, cmap='Blues', cbar=False, 
                linewidths=0.5, linecolor='lightgray')
    
    # Add TSS values as text annotations
    for i, tss in enumerate(df['TSS']):
        plt.text(len(feature_names) + 0.5, i + 0.5, f"{tss:.3f}", 
                 ha='center', va='center', fontsize=11)
    
    # Add TSS header
    plt.text(len(feature_names) + 0.5, -0.1, 'TSS', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Set y-tick labels
    config_labels = []
    for i, idx in enumerate(df.index):
        result = results[idx]
        if result['config_name'] == 'Full model':
            config_labels.append('Full model')
        else:
            config_labels.append(result['description'])
    
    plt.yticks(np.arange(len(config_labels)) + 0.5, config_labels, fontsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "ablation_feature_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    
    return output_file

def analyze_ablation_impact(results):
    """Analyze and report the impact of each ablation"""
    # Find the full model result
    full_model = None
    for result in results:
        if result['config_name'] == 'Full model':
            full_model = result
            break
    
    if not full_model:
        print("Full model configuration not found in results")
        return
    
    # Calculate impact of each ablation
    print("\nImpact Analysis:")
    print(f"Full model TSS: {full_model['tss']:.3f}")
    
    for result in results:
        if result['config_name'] != 'Full model':
            impact = full_model['tss'] - result['tss']
            impact_percent = (impact / full_model['tss']) * 100
            print(f"{result['description']}: -{impact:.3f} TSS (-{impact_percent:.1f}%)")

if __name__ == "__main__":
    # Check if results file exists
    results_path = "results/ablation/ablation_results_24h_M_class.json"
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run ablation_study.py first")
        exit(1)
    
    # Load results
    results = load_ablation_results(results_path)
    
    # Create visualizations
    create_bar_chart(results)
    create_heatmap(results)
    
    # Create table
    table_df = create_comparison_table(results)
    
    # Analyze impact
    analyze_ablation_impact(results) 