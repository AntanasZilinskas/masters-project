'''
 author: Antanas Zilinskas

 Analysis of Sequence Length Ablation Study Results

 This script reads all result files from the sequence length ablation study,
 finds the optimal sequence length for each flare class and time window,
 and presents the findings.
'''

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def find_result_files():
    """Find all result JSON files in the results directory."""
    results_dir = os.path.join("results")
    if not os.path.exists(results_dir):
        # Try parent directory
        results_dir = os.path.join("..", "results")

    pattern = os.path.join(results_dir, "seq_length_ablation_*.json")
    return glob.glob(pattern)


def load_result_file(file_path):
    """Load a result file and return the parsed JSON."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def find_best_sequence_length(metrics, metric_name="TSS"):
    """Find the sequence length that maximizes the given metric."""
    if not metrics:
        return None, 0.0

    best_length = None
    best_value = -1

    for length, values in metrics.items():
        if metric_name in values and values[metric_name] != "N/A":
            if best_length is None or values[metric_name] > best_value:
                best_length = int(length)
                best_value = values[metric_name]

    return best_length, best_value


def analyze_results():
    """Analyze all ablation study results and summarize findings."""
    result_files = find_result_files()
    if not result_files:
        print("No result files found!")
        return

    print(f"Found {len(result_files)} result files to analyze.")

    # Store best values
    best_lengths = {}
    metrics_by_config = defaultdict(dict)
    all_metrics = defaultdict(list)

    # Extract metrics from all result files
    for file_path in result_files:
        result = load_result_file(file_path)
        if not result:
            continue

        flare_class = result.get('flare_class')
        time_window = result.get('time_window')
        metrics = result.get('metrics', {})

        if not flare_class or not time_window or not metrics:
            continue

        # Store results in organized structure
        key = f"{flare_class}_{time_window}"

        # Find best length for each metric
        for metric in [
            'accuracy',
            'precision',
            'recall',
            'balanced_accuracy',
                'TSS']:
            best_length, best_value = find_best_sequence_length(
                metrics, metric)
            if best_length is not None:
                metrics_by_config[key][metric] = {
                    'best_length': best_length,
                    'best_value': best_value
                }
                all_metrics[metric].append({
                    'flare_class': flare_class,
                    'time_window': time_window,
                    'best_length': best_length,
                    'best_value': best_value
                })

        # Find best overall length (using TSS)
        best_length, best_value = find_best_sequence_length(metrics, 'TSS')
        if best_length is not None:
            best_lengths[key] = {
                'flare_class': flare_class,
                'time_window': time_window,
                'best_length': best_length,
                'TSS': best_value
            }

    # Create output directory
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Generate summary report
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("Sequence Length Ablation Study Analysis\n")
        f.write("=====================================\n\n")

        f.write("Best Sequence Lengths (by TSS):\n")
        f.write("----------------------------\n")
        for key, values in sorted(best_lengths.items()):
            f.write(
                f"{values['flare_class']}-class, {values['time_window']}h window: "
                f"{values['best_length']} timesteps (TSS: {values['TSS']:.4f})\n")

        f.write("\n\nMetric-Specific Best Lengths:\n")
        f.write("--------------------------\n")
        for key, metrics in sorted(metrics_by_config.items()):
            f.write(f"\n{key}:\n")
            for metric, values in metrics.items():
                f.write(f"  {metric}: {values['best_length']} timesteps "
                        f"({values['best_value']:.4f})\n")

    print(
        f"Generated summary report at {os.path.join(output_dir, 'summary.txt')}")

    # Create visualizations

    # 1. Bar chart of best sequence length by flare class and time window
    fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)

    data = pd.DataFrame(list(best_lengths.values()))

    # Sort by flare class and time window for better visualization
    flare_order = ['C', 'M', 'M5']
    time_window_order = ['24', '48', '72']

    data['flare_class'] = pd.Categorical(
        data['flare_class'],
        categories=flare_order,
        ordered=True)
    data['time_window'] = pd.Categorical(
        data['time_window'],
        categories=time_window_order,
        ordered=True)
    data = data.sort_values(['flare_class', 'time_window'])

    # Create labels for x-axis
    data['label'] = data.apply(
        lambda row: f"{row['flare_class']}-{row['time_window']}h", axis=1)

    # Create the bar chart
    bars = ax.bar(data['label'], data['best_length'], color='skyblue')

    # Add the TSS values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'TSS: {data.iloc[i]["TSS"]:.3f}',
                ha='center', va='bottom', rotation=0, fontsize=8)

    plt.title('Optimal Sequence Length by Flare Class and Time Window')
    plt.xlabel('Flare Class - Time Window')
    plt.ylabel('Optimal Sequence Length')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            'optimal_sequence_lengths.png'),
        dpi=300)
    plt.close()

    # 2. Heatmap of best metric values
    for metric in ['accuracy', 'precision', 'recall', 'TSS']:
        data = pd.DataFrame(all_metrics[metric])

        # Create pivot table
        pivot = data.pivot(
            index='flare_class',
            columns='time_window',
            values='best_value')

        # Sort rows and columns
        pivot = pivot.reindex(index=flare_order)
        pivot = pivot.reindex(columns=time_window_order)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot,
            annot=True,
            cmap='viridis',
            fmt='.4f',
            vmin=0,
            vmax=1)
        plt.title(f'Best {metric.capitalize()} by Flare Class and Time Window')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'best_{metric}.png'), dpi=300)
        plt.close()

    print(f"Generated visualizations in {output_dir} directory")


if __name__ == "__main__":
    analyze_results()
