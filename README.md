# Masters Project: Solar Flare Prediction Model

This repository contains code for solar flare prediction models using transformer-based architectures.

## Repository Structure

```
masters-project/
├── config/                    # Configuration files
├── data/                      # Data files
├── docs/                      # Documentation
├── logs/                      # Training logs
├── models/                    # Model definitions
├── nature_models/             # Models for Nature dataset
├── results/                   # Results and visualizations
├── scripts/                   # Utility scripts
├── tests/                     # Test scripts
├── utils/                     # Utilities
└── weights/                   # Model weights and tracking
```

## Key Files

- `models/SolarKnowledge_model.py` - The main solar flare prediction model
- `models/SolarKnowledge_run_all_trainings.py` - Script to train models
- `models/SolarKnowledge_run_all_tests.py` - Script to test models
- `scripts/model_tracker.py` - Track and compare model performance
- `docs/README_model_tracker.md` - Documentation for model tracking

## Usage

To train a model:
```bash
python models/SolarKnowledge_run_all_trainings.py -d "Model description"
```

To test a model:
```bash
python models/SolarKnowledge_run_all_tests.py -t [TIMESTAMP]
```

To track and compare models:
```bash
python scripts/model_tracker.py scan
python scripts/model_tracker.py list
python scripts/model_tracker.py compare -m TSS
```

# Visualisation Resources

https://spacebook.com/explorer

https://cesium.com/blog/2024/11/26/spacebook-makes-space-data-accessible-for-all-with-cesiumjs/

# Data source

https://jsoc1.stanford.edu/data/hmi/fits/2022/01/01/

https://sdo.gsfc.nasa.gov/data/dataaccess.php

