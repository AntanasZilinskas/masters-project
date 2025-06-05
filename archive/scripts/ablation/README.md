# Ablation Study for Solar Flare Prediction Models

This directory contains scripts for conducting and visualizing ablation studies on the SolarKnowledge model architecture.

## Overview

Ablation studies help understand the contribution of different components within the model architecture by systematically removing or modifying specific components and measuring the impact on performance.

## Scripts

1. **ablation_study.py** - Conducts the actual ablation study experiments
2. **visualize_ablation.py** - Creates visualizations and formatted output from ablation results

## Configurations Tested

The ablation study tests the following configurations:

- **Full model**: (Conv1D + BN) + LSTM + 4 TEBs + heavy dropout
- **No LSTM**: only conv + BN, then TEBs
- **No conv**: BN then LSTM
- **Reduced TEBs**: 2 layers instead of 4
- **No class weighting**: No class weights for imbalanced data
- **Light dropout**: dropout = 0.1 (lighter)

## Usage

### Running the Ablation Study

```bash
# Make sure you're in the project root directory
cd /path/to/masters-project

# Run the ablation study (will take several hours)
python scripts/ablation/ablation_study.py
```

This will:
1. Train models with each configuration on the 24h M-class prediction task
2. Evaluate TSS performance on the test set
3. Save results to `results/ablation/ablation_results_24h_M_class.json`
4. Print a summary of results in both plain text and LaTeX table formats

### Visualizing the Results

After running the ablation study, you can visualize the results:

```bash
# Create visualizations from the results
python scripts/ablation/visualize_ablation.py
```

This will:
1. Create a bar chart comparing TSS values across configurations
2. Generate a heatmap showing the presence/absence of features across configurations
3. Print the results as a formatted table including LaTeX code for your paper
4. Analyze the impact of each ablation relative to the full model

## Output Files

- **ablation_results_24h_M_class.json** - Raw results data
- **ablation_tss_chart.png** - Bar chart visualization of TSS values
- **ablation_feature_heatmap.png** - Heatmap showing feature usage across configurations

## Extending the Study

To test additional configurations, modify the `CONFIGURATIONS` list in `ablation_study.py` with new settings.
