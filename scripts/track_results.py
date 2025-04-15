import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_results(filepath="models/this_work_results.json"):
    """Load results from the JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}

def generate_summary(results):
    """Generate a text summary of the results."""
    if not results:
        return "No results found."
    
    summary = "=== SOLAR FLARE PREDICTION RESULTS SUMMARY ===\n"
    summary += f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for horizon in sorted(results.keys()):
        summary += f"TIME HORIZON: {horizon} hours\n"
        summary += "=" * 40 + "\n"
        
        for flare_class in sorted(results[horizon].keys()):
            metrics = results[horizon][flare_class]
            summary += f"  Class {flare_class}:\n"
            for metric, value in metrics.items():
                summary += f"    {metric}: {value:.4f}\n"
            summary += "\n"
    
    return summary

def plot_results(results, metric="TSS", save_path="results/results_plot.png"):
    """Create a visualization of the results focusing on the given metric."""
    if not results:
        print("No results to plot.")
        return
    
    horizons = sorted(results.keys())
    flare_classes = sorted(list(results[horizons[0]].keys()))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    opacity = 0.8
    index = np.arange(len(horizons))
    
    for i, flare_class in enumerate(flare_classes):
        metric_values = [results[horizon][flare_class][metric] for horizon in horizons]
        plt.bar(index + i*bar_width, metric_values, bar_width,
                alpha=opacity, label=f'Class {flare_class}')
    
    plt.xlabel('Prediction Horizon (hours)')
    plt.ylabel(metric)
    plt.title(f'Solar Flare Prediction Performance ({metric})')
    plt.xticks(index + bar_width, horizons)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
def create_dashboard():
    """Create a comprehensive dashboard of results."""
    results = load_results()
    if not results:
        return
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Generate and print summary
    summary = generate_summary(results)
    print(summary)
    
    # Save summary to file
    with open("results/results_summary.txt", "w") as f:
        f.write(summary)
    
    # Create visualizations for different metrics
    for metric in ["TSS", "accuracy", "precision", "recall"]:
        plot_results(results, metric=metric, save_path=f"results/results_{metric}.png")
    
    print("Dashboard generated successfully!")
    print("Summary saved to results/results_summary.txt")
    print("Plots saved as results/results_*.png")

if __name__ == "__main__":
    create_dashboard() 