# SHARP ML Dataset Pipeline

A reproducible data pipeline for creating a machine learning dataset from SDO/HMI SHARP data.

## Raw Sources

- **SDO/HMI SHARP CEA 720s**: Public domain data, courtesy of NASA/SDO and the HMI science team
- **NOAA GOES X-ray flare list**: Public domain data from NOAA Space Weather Prediction Center (ftp://ftp.swpc.noaa.gov/pub/indices/events/)

## Pipeline Overview

This pipeline creates a fully reproducible ML dataset for solar flare prediction using the SHARP (Space-weather HMI Active Region Patches) parameters and NOAA flare event data.

The novelty of this dataset comes from:
- Transparent time-slicing and labeling methodology
- Additional engineered features (temporal derivatives, rolling means, flare history)
- A fully deterministic and scripted pipeline

## Steps to Reproduce

To recreate the dataset from scratch:

```bash
# Download raw SHARP data from JSOC
bash 00_download_raw.sh

# Clean and merge SHARP patch summary files
python 01_clean_and_merge.py

# Add flare labels from NOAA event list (default 24h prediction horizon)
python 02_add_flare_labels.py --horizon 24

# Engineer additional features
python 03_engineer_features.py

# Create train/validation/test splits
python 04_make_splits.py
```

The pipeline will create the following files:
- `data/all_sharp_10min.csv.gz`: Cleaned and merged SHARP parameters
- `data/sharp_with_labels.csv.gz`: SHARP data with flare labels
- `data/sharp_features_v1.csv.gz`: Complete dataset with engineered features
- `data/splits/train.csv`: Training split (2010-2018)
- `data/splits/val.csv`: Validation split (2019-2021)
- `data/splits/test.csv`: Test split (2022-2024)
- `data/splits/meta.json`: Metadata including row counts and file hashes

## Dataset Features

The dataset includes the standard SHARP parameters used in solar flare prediction literature:

- **Magnetic field parameters**: USFLUX, MEANGBT, MEANGBZ, MEANGBH
- **Electric current parameters**: TOTUSJZ, MEANJZH, TOTUSJH, ABSNJZH
- **Helicity and Lorentz force parameters**: MEANALP, MEANSHR, SHRGT45
- **Energy parameters**: SAVNCPP, MEANPOT, TOTPOT
- **Inversion parameter**: R_VALUE

In addition, we introduce new engineered features:

- **dUSFLUX_dt**: Time derivative of USFLUX over 1 hour
- **R_VALUE_rolling3h_mean**: 3-hour rolling mean of R_VALUE
- **TOTUSJH_rolling3h_mean**: 3-hour rolling mean of TOTUSJH
- **USFLUX_rolling3h_mean**: 3-hour rolling mean of USFLUX
- **flare_count_48h**: Count of flares in the same active region in past 48h
- **cos_heliographic_lat**: Cosine of heliographic latitude (for line-of-sight bias correction)

## Label Information

The dataset includes binary labels for flare prediction at different thresholds:

- **flare_C_24h**: C-class or stronger flare within 24 hours
- **flare_M_24h**: M-class or stronger flare within 24 hours
- **flare_M5_24h**: M5-class or stronger flare within 24 hours
- **flare_X_24h**: X-class flare within 24 hours

## Dataset Splits

The data is split chronologically to avoid temporal leakage:

- **Train**: 2010-2018
- **Validation**: 2019-2021
- **Test**: 2022-2024

## Data Processing Details

- TAI timestamps are converted to UTC ISO8601 format
- Rows with NaN or zero values in required features are dropped
- The pipeline differs from Abdullah et al.'s approach, which fills missing values with zeros

## Publication

To publish the processed dataset:

1. Zip only the `data/splits/` directory
2. Upload to Zenodo as "Derived Data – SHARP ML Split (Zilinskas 2025)"
3. License: CC-BY-4.0 (covers only derived tables, raw data remains public domain)
4. In "Related identifiers" add references to:
   - JSOC collection handle
   - NOAA event list URL

## Citation

If you use this dataset in your research, please cite it as:

```
Zilinskas, A. (2025). Derived Data - SHARP ML Split. [Data set]. Zenodo. https://doi.org/[DOI]
```

## License

The derived dataset is released under CC-BY-4.0.
The pipeline code is released under the MIT License.
Raw solar data from NASA/SDO and NOAA are in the public domain.
