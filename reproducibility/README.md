# EVEREST: Extreme EVector-based Representation and Environment Stability Transformer

This repository contains code for the EVEREST model, a transformer-based architecture for time series prediction and anomaly detection.

## Repository Structure

```
.
├── data/
│   ├── solar_data/   # Solar flare prediction data (included)
│   │   ├── training_data_*.csv  # Training datasets for different flare classes and time windows
│   │   ├── testing_data_*.csv   # Testing datasets for different flare classes and time windows
│   │   ├── solar_events.csv     # Solar event metadata
│   │   └── harp_to_noaa.csv     # Mapping between HARP and NOAA regions
│   └── SKAB/         # Industrial anomaly detection data (included)
│       ├── training_data_*.csv  # Training datasets for different valve scenarios
│       └── testing_data_*.csv   # Testing datasets for different valve scenarios
├── models/
│   ├── dataset_analysis/
│   │   └── model_comparison.json  # Comparison with other models
│   ├── everest.py                 # Core EVEREST model implementation
│   ├── model_tracking.py          # Utilities for tracking models and metrics
│   ├── utils.py                   # General utility functions
│   ├── train.py                   # Training script for solar flare prediction
│   ├── evaluate_solar.py          # Evaluation script for solar flare prediction
│   ├── train_skab.py              # Training script for SKAB anomaly detection
│   └── evaluate_skab.py           # Evaluation script for SKAB anomaly detection
└── README.md                      # This file
```

## Experiments

This repository includes code for two primary experiments:

1. **Solar Flare Prediction**: Using the EVEREST model to predict solar flares based on magnetogram data.
2. **Industrial Anomaly Detection**: Applying the EVEREST model to the SKAB (Skoltech Anomaly Benchmark) dataset to detect valve anomalies.

## Data Processing

### Solar Flare Data

The solar flare prediction experiment uses SHARP (Space-weather HMI Active Region Patch) parameters from the SDO/HMI instrument. The data processing pipeline includes:

1. **Feature Selection**: We use 9 SHARP parameters that have been shown to correlate with flare activity.
2. **Sequence Creation**: Each data point consists of 24 timesteps (1 hour cadence) for each feature.
3. **Labeling**: Data is labeled based on the occurrence of M/X-class flares within the prediction window (24, 48, or 72 hours).
4. **Train/Test Split**: The data is split chronologically to prevent data leakage.
5. **Normalization**: Features are normalized using standard scaling (zero mean, unit variance).
6. **Class Balancing**: Due to the rarity of flare events, the dataset is balanced through undersampling of the negative class.

The processed solar flare data is included in the repository under `data/solar_data/` and includes:
- Datasets for C, M, and M5+ class flares
- 24, 48, and 72-hour prediction windows
- Properly split training and testing sets
- All required metadata and mappings

### SKAB Data

The SKAB (Skoltech Anomaly Benchmark) dataset contains industrial sensor readings from valve systems with labeled anomalies. Our processing approach includes:

1. **Sequence Creation**: We use full 24-timestep sequences instead of truncation.
2. **Feature Engineering**: 
   - All 16 original sensor readings are preserved
   - Velocity features (first-order differences) are added to capture rate of change
3. **Normalization**: StandardScaler is applied instead of MinMaxScaler for better generalization.
4. **Overlapping Windows**: We use a stride of 2 for overlapping windows during training.
5. **Chronological Ordering**: Train/test splits preserve the temporal ordering of events.
6. **Unified Model**: A single model is trained on all valve scenarios simultaneously, improving generalization.

The processed SKAB data is included in the repository under `data/SKAB/` and includes:
- Training and testing datasets for all valve scenarios (valve1_0, valve1_1, valve1_10, valve1_11, valve1_12, valve2_0, valve2_1)
- Normal (anomaly-free) data
- All datasets are ready to use with the provided training scripts

## Running the Code

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib, Scikit-learn
- Other dependencies listed in requirements.txt

Install dependencies:
```
pip install -r requirements.txt
```

### Training the Models

For the solar flare prediction task:
```
cd models
python train.py
```

For the SKAB anomaly detection task:
```
cd models
python train_skab.py
```

### Evaluating the Models

For solar flare prediction:
```
cd models
python evaluate_solar.py
```

For SKAB anomaly detection:
```
cd models
python evaluate_skab.py
```

## Results

### Solar Flare Prediction

The EVEREST model achieves competitive performance on solar flare prediction:
- TSS (True Skill Statistic): 0.60-0.75 (depending on flare class and prediction window)
- HSS (Heidke Skill Score): 0.45-0.60
- Balanced accuracy: 75-85%

### SKAB Anomaly Detection

The model achieves state-of-the-art results on the SKAB dataset:
- Overall F1 score: 98.16%
- Accuracy: 98.20%
- TSS: 0.964
- Precision: 97.56%
- Recall: 98.76%

This significantly outperforms previous approaches (best previous F1: 91-96%).