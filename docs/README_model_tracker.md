# Solar Flare Prediction Model Tracker

A command-line tool for tracking and comparing solar flare prediction model results by scanning model metadata files.

## Features

- Automatically scan and consolidate model metadata from trained models
- View detailed model information including architecture, training parameters, and results
- List all tracked models with their metadata
- Compare models across specific metrics with optional filtering by flare class and time horizon
- Export all model details to CSV for further analysis
- Generate summary statistics across all models

## Integrated Workflow

This system provides a complete workflow for tracking model performance:

1. When you train models with `SolarKnowledge_run_all_trainings.py`, timestamped model metadata is automatically saved
2. When you test models with `SolarKnowledge_run_all_tests.py`, test results are added to the metadata
3. Use `model_tracker.py` to scan, analyze, and compare all your models' results

## Folder Structure

The system uses the following directory structure:
```
weights/
  24/              # Time window
    C/             # Flare class
      model_weights.weights.h5            # Latest model weights
      model_weights_20250415_123456.h5    # Timestamped model weights
      metadata_latest.json                # Latest metadata
      metadata_20250415_123456.json       # Timestamped metadata
    M/
    M5/
  48/
    C/
    M/
    M5/
  72/
    C/
    M/
    M5/
```

## Usage

### Scan for Model Metadata

Scan the weights directory for model metadata files and consolidate them:

```bash
python model_tracker.py scan
```

This will search in the weights directory structure for all timestamped metadata files and consolidate them into the tracking system.

### List All Tracked Models

```bash
python model_tracker.py list
```

This displays all models with their flare class, time window, timestamp, and description.

### View Specific Model Results

```bash
python model_tracker.py show MODEL_TIMESTAMP
```

This will display detailed results for the specified model, including:
- Model architecture details (transformer blocks, embedding dimensions, etc.)
- Training information (optimizer, learning rate, epochs, etc.)
- Test results for all metrics

### Compare Models

Compare models on a specific metric with optional filtering:

```bash
python model_tracker.py compare -m TSS -c M -t 24
```

Options:
- `-m, --metric`: Metric to compare (default: TSS)
- `-c, --class`: Filter by flare class (optional)
- `-t, --horizon`: Filter by time horizon in hours (optional)

The comparison will display:
- Results sorted by performance
- Summary statistics (best, worst, mean, median, standard deviation)

### Export to CSV

Export all model details to a CSV file for further analysis:

```bash
python model_tracker.py export -o my_results.csv
```

Options:
- `-o, --output`: Output CSV file (default: model_comparison.csv)

## Examples

### Typical Workflow

1. Train your models:
   ```bash
   python models/SolarKnowledge_run_all_trainings.py -d "Model with 6 transformer blocks"
   ```

2. Test your models (note the timestamp from training output):
   ```bash
   python models/SolarKnowledge_run_all_tests.py -t 20250415_123456
   ```

3. Scan model metadata and analyze results:
   ```bash
   python model_tracker.py scan
   python model_tracker.py list
   python model_tracker.py compare -m TSS
   ```

### Viewing Model Architecture Details

To see detailed information about a specific model:

```bash
python model_tracker.py show 20250415_123456
```

## Metadata Storage

All model metadata is stored in timestamped files within the weights directory structure:
- Each model saves a `metadata_[timestamp].json` file in its respective directory
- The model tracker scans and consolidates these files into `model_results.json`
