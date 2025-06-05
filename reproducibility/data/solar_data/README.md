# Solar Flare Prediction Data

This directory contains the processed solar flare prediction data used in the EVEREST model experiments.

## Dataset Structure

```
solar_data/
├── training_data_C_24.csv    # Training data for C-class flares with 24-hour prediction window
├── testing_data_C_24.csv     # Test data for C-class flares with 24-hour prediction window
├── training_data_C_48.csv    # Training data for C-class flares with 48-hour prediction window
├── testing_data_C_48.csv     # Test data for C-class flares with 48-hour prediction window
├── training_data_C_72.csv    # Training data for C-class flares with 72-hour prediction window
├── testing_data_C_72.csv     # Test data for C-class flares with 72-hour prediction window
├── training_data_M_24.csv    # Training data for M-class flares with 24-hour prediction window
├── testing_data_M_24.csv     # Test data for M-class flares with 24-hour prediction window
├── training_data_M_48.csv    # Training data for M-class flares with 48-hour prediction window
├── testing_data_M_48.csv     # Test data for M-class flares with 48-hour prediction window
├── training_data_M_72.csv    # Training data for M-class flares with 72-hour prediction window
├── testing_data_M_72.csv     # Test data for M-class flares with 72-hour prediction window
├── training_data_M5_24.csv   # Training data for M5+-class flares with 24-hour prediction window
├── testing_data_M5_24.csv    # Test data for M5+-class flares with 24-hour prediction window
├── training_data_M5_48.csv   # Training data for M5+-class flares with 48-hour prediction window
├── testing_data_M5_48.csv    # Test data for M5+-class flares with 48-hour prediction window
├── training_data_M5_72.csv   # Training data for M5+-class flares with 72-hour prediction window
├── testing_data_M5_72.csv    # Test data for M5+-class flares with 72-hour prediction window
├── solar_events.csv          # Solar event metadata including flare information
└── harp_to_noaa.csv          # Mapping between HARP and NOAA active region numbers
```

## Downloading the Data

The complete dataset is available through an anonymous Google Drive link:

[Solar Flare Prediction Dataset](https://drive.google.com/drive/folders/1ayWbjzVBAym7exag9TImsp3fjEFwm9LA?usp=share_link)

To download the data:

1. Visit the Google Drive link above
2. You can download individual files by right-clicking on them and selecting "Download"
3. To download all files at once:
   - Select all files (Ctrl+A or Cmd+A)
   - Right-click and select "Download"
   - Google Drive will create a zip file containing all the data

4. Extract the downloaded files into this directory (`solar_data/`)

Note that these files are large (especially the training datasets, ~300MB each), so the download may take some time depending on your internet connection.

### Command-line Download Option

You can also download the files using the command line with the `gdown` Python package:

```bash
# Install gdown if you don't have it
pip install gdown

# Change to the solar_data directory
cd /path/to/everest_code/data/solar_data

# Download files using the folder ID
gdown --folder https://drive.google.com/drive/folders/1ayWbjzVBAym7exag9TImsp3fjEFwm9LA --remaining-ok
```

If you prefer to download individual files, you can use the file ID from the Google Drive link:

```bash
# Example for downloading a specific file (harp_to_noaa.csv)
# First get the file ID from the Google Drive URL when you click on the file
gdown FILE_ID -O harp_to_noaa.csv
```

## Data Processing Details

These files were processed with the following steps:

1. **Feature Selection**: 9 SHARP parameters that correlate with flare activity were selected
2. **Sequence Creation**: 24-timestep sequences (1 hour cadence) were extracted
3. **Labeling**: Data was labeled based on flare occurrence within the prediction window
4. **Train/Test Split**: Chronological splitting was used to prevent data leakage
5. **Normalization**: StandardScaler was applied to normalize the features
6. **Class Balancing**: The negative class was undersampled to address class imbalance

## File Format

Each CSV file contains the following columns:
- `HARPNUM`: Unique identifier for each active region
- `step`: Timestep within the sequence (0-23)
- `timestamp`: Observation timestamp
- `class`: 'F' for flare, 'N' for non-flare
- Feature columns: 9 SHARP parameters

## Usage

These files can be used directly with the training and evaluation scripts:

```python
# Example code to load the data
import pandas as pd

# Load training data for M-class flares with 24-hour prediction window
df = pd.read_csv('training_data_M_24.csv')

# Group by HARPNUM to get sequences
sequences = df.groupby('HARPNUM')

# Extract features and labels
X = []
y = []

for harpnum, group in sequences:
    # Sort by step to ensure correct sequence
    group = group.sort_values('step')
    
    # Extract features (all columns except metadata)
    feature_cols = [col for col in df.columns 
                  if col not in ['class', 'timestamp', 'step', 'HARPNUM']]
    seq = group[feature_cols].values
    
    # Get label (same for all rows in the group)
    label = 1 if group['class'].iloc[0] == 'F' else 0
    
    X.append(seq)
    y.append(label)
``` 