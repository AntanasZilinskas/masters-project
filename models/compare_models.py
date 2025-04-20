"""
Compare the performance of the original SolarKnowledge model and the multimodal version.

Author: Antanas Zilinskas
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(filename):
    """Load results from a JSON file."""
    if not os.path.exists(filename):
        print(f"Results file {filename} not found.")
        return None

    with open(filename, 'r') as f:
        return json.load(f)


def plot_comparison(original_results, multimodal_results, metric='TSS'):
    """Plot a comparison of the original and multimodal models."""
    if original_results is None or multimodal_results is None:
        print("Cannot plot comparison: missing results.")
        return

    # Time windows and flare classes
    time_windows = ['24', '48', '72']
    flare_classes = ['C', 'M', 'M5']

    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bar_width = 0.35
    index = np.arange(len(flare_classes))

    for i, time_window in enumerate(time_windows):
        original_values = []
        multimodal_values = []

        for flare_class in flare_classes:
            # Get the metric value for the original model
            if time_window in original_results and flare_class in original_results[time_window]:
                orig_value = original_results[time_window][flare_class].get(
                    metric, 'N/A')
                if orig_value == 'N/A':
                    orig_value = 0
            else:
                orig_value = 0

            # Get the metric value for the multimodal model
            if time_window in multimodal_results and flare_class in multimodal_results[
                    time_window]:
                multi_value = multimodal_results[time_window][flare_class].get(
                    metric, 'N/A')
                if multi_value == 'N/A':
                    multi_value = 0
            else:
                multi_value = 0

            original_values.append(orig_value)
            multimodal_values.append(multi_value)

        # Plot the bars
        axes[i].bar(index, original_values, bar_width, label='Original')
        axes[i].bar(
            index + bar_width,
            multimodal_values,
            bar_width,
            label='Multimodal')

        # Add labels and title
        axes[i].set_xlabel('Flare Class')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'{time_window}h Forecast Window')
        axes[i].set_xticks(index + bar_width / 2)
        axes[i].set_xticklabels(flare_classes)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'model_comparison_{metric}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # Load results
    original_results = load_results('this_work_results.json')
    multimodal_results = load_results('multimodal_results.json')

    # Plot comparisons for different metrics
    for metric in [
        'TSS',
        'accuracy',
        'precision',
        'recall',
            'balanced_accuracy']:
        plot_comparison(original_results, multimodal_results, metric)
