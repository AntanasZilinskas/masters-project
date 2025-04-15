# Solar Flare Prediction Results Tracker

This toolkit helps you track and visualize model performance for your solar flare prediction models.

## Quick Start

1. Run the dashboard generator:
   ```
   python track_results.py
   ```

2. The script will:
   - Generate a text summary in `results_summary.txt`
   - Create visualization plots for key metrics (TSS, accuracy, precision, recall)
   - Display results in the console

## Adding New Results

When you have new model results:

1. Update the `models/this_work_results.json` file with your new results
2. Run the tracker again to generate updated visualizations

## JSON Structure

The tracker expects your results in the following format:

```json
{
    "24": {  // Prediction horizon in hours
        "C": {  // Flare class
            "accuracy": 0.9871,
            "precision": 0.989,
            "recall": 0.9921,
            "balanced_accuracy": 0.9843,
            "TSS": 0.9687
        },
        "M": {
            // metrics
        }
    },
    "48": {
        // Similar structure as above
    }
}
```

## Extending Prediction Horizons

If you decide to test longer prediction horizons (e.g., 96hr, 120hr), simply add them to your JSON results file using the same structure, and the tracker will automatically include them in the visualizations.

## Customizing Visualizations

You can modify the `track_results.py` script to:
- Change which metrics are plotted
- Adjust plot appearance
- Add additional analysis metrics 