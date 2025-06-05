# SKAB Dataset

This directory contains the processed SKAB (Skoltech Anomaly Benchmark) dataset used in the EVEREST model experiments.

The original dataset was obtained from [https://github.com/waico/SKAB](https://github.com/waico/SKAB) and processed according to the procedures described in the main README.

## Dataset Structure

```
SKAB/
├── training_data_valve1_0.csv     # Training data for valve1, experiment 0
├── testing_data_valve1_0.csv      # Testing data for valve1, experiment 0
├── training_data_valve1_1.csv     # Training data for valve1, experiment 1
├── testing_data_valve1_1.csv      # Testing data for valve1, experiment 1
├── training_data_valve1_10.csv    # Training data for valve1, experiment 10
├── testing_data_valve1_10.csv     # Testing data for valve1, experiment 10
├── training_data_valve1_11.csv    # Training data for valve1, experiment 11
├── testing_data_valve1_11.csv     # Testing data for valve1, experiment 11
├── training_data_valve1_12.csv    # Training data for valve1, experiment 12
├── testing_data_valve1_12.csv     # Testing data for valve1, experiment 12
├── training_data_valve2_0.csv     # Training data for valve2, experiment 0
├── testing_data_valve2_0.csv      # Testing data for valve2, experiment 0
├── training_data_valve2_1.csv     # Training data for valve2, experiment 1
├── testing_data_valve2_1.csv      # Testing data for valve2, experiment 1
├── training_data_normal_anomaly-free.csv  # Training data for normal (no anomalies)
└── testing_data_normal_anomaly-free.csv   # Testing data for normal (no anomalies)
```

## Downloading the Data

The complete SKAB dataset is available through an anonymous Google Drive link:

[SKAB Dataset](https://drive.google.com/drive/folders/16FAVrmquq4NrB_LJW1-OywhSGQryL94F?usp=share_link)

To download the data:

1. Visit the Google Drive link above
2. You can download individual files by right-clicking on them and selecting "Download"
3. To download all files at once:
   - Select all files (Ctrl+A or Cmd+A)
   - Right-click and select "Download"
   - Google Drive will create a zip file containing all the data

4. Extract the downloaded files into this directory (`SKAB/`)

Note that these files are smaller than the solar flare datasets but still may take some time to download depending on your internet connection.

### Command-line Download Option

You can also download the files using the command line with the `gdown` Python package:

```bash
# Install gdown if you don't have it
pip install gdown

# Change to the SKAB directory
cd /path/to/everest_code/data/SKAB

# Download files using the folder ID
gdown --folder https://drive.google.com/drive/folders/16FAVrmquq4NrB_LJW1-OywhSGQryL94F --remaining-ok
```

If you prefer to download individual files, you can use the file ID from the Google Drive link:

```bash
# Example for downloading a specific file (testing_data_valve1_0.csv)
# First get the file ID from the Google Drive URL when you click on the file
gdown FILE_ID -O testing_data_valve1_0.csv
```

## Data Processing Details

These files were processed with the following steps:

1. **Sequence Creation**: Full 24-timestep sequences were extracted from the raw data
2. **Feature Engineering**: 
   - All 16 original sensor readings were preserved
   - Velocity features (first-order differences) were added to capture rate of change
3. **Normalization**: StandardScaler was applied to normalize the data
4. **Overlapping Windows**: A stride of 2 was used for overlapping windows during training
5. **Chronological Ordering**: Train/test splits preserve the temporal ordering of events

## File Format

Each CSV file contains the following columns:
- `HARPNUM`: Unique identifier for each sequence (NOTE: This is just a sequence ID and does not refer to solar active regions as in the solar flare dataset. The name is reused for code compatibility.)
- `step`: Timestep within the sequence (0-23)
- `timestamp`: Original timestamp from the SKAB dataset
- `class`: 'N' for normal, 'F' for fault/anomaly
- `sensor_1` through `sensor_16`: Original sensor readings
- `velocity_1` through `velocity_16`: First-order differences of sensor readings

## Usage

These files can be used directly with the training and evaluation scripts:

```python
# Example code to load the data
import pandas as pd

# Load training data for valve1_0
df = pd.read_csv('training_data_valve1_0.csv')

# Group by HARPNUM to get sequences
sequences = df.groupby('HARPNUM')

# Extract features and labels
X = []
y = []

for harpnum, group in sequences:
    # Sort by step to ensure correct sequence
    group = group.sort_values('step')
    
    # Extract features
    feature_cols = [col for col in df.columns 
                  if col not in ['class', 'timestamp', 'step', 'HARPNUM']]
    seq = group[feature_cols].values
    
    # Get label (same for all rows in the group)
    label = 1 if group['class'].iloc[0] == 'F' else 0
    
    X.append(seq)
    y.append(label) 