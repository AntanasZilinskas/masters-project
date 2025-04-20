# Masters Project: Solar Flare Prediction Model

![CI](https://github.com/antanaszilinskas/masters-project/actions/workflows/ci.yml/badge.svg)
![Nightly Eval](https://github.com/antanaszilinskas/masters-project/actions/workflows/nightly.yml/badge.svg)
![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.0000000.svg)

This repository contains code for solar flare prediction models using transformer-based architectures.

## Repository Structure

```
masters-project/
├── .github/                   # GitHub workflows and config
├── config/                    # Configuration files
├── datasets/                  # Data files
│   └── tiny_sample/           # Small dataset for CI
├── docker/                    # Docker configurations
├── docs/                      # Documentation
├── logs/                      # Training logs
├── models/                    # Model definitions
├── results/                   # Results and visualizations
├── scripts/                   # Utility scripts
├── solar_knowledge/           # Core package code
├── tests/                     # Test scripts
└── utils/                     # Utilities
```

## Continuous Integration & Deployment

This project uses GitHub Actions for CI/CD:

- **CI Pipeline**: Runs unit tests, linting, and smoke training on every push and PR
- **Nightly Evaluation**: Runs full model evaluation on recent data every day
- **Documentation**: Automatically publishes updated API documentation

## Usage

### Installation

```bash
# For CPU usage
pip install -r requirements.txt

# For GPU usage with conda
conda env create -f environment.yml
conda activate sk-env
```

### Docker

```bash
docker build -t solar-knowledge -f docker/Dockerfile .
docker run -it solar-knowledge
```

### Training

```bash
python -m solar_knowledge.smoke_train --data-dir datasets/tiny_sample --epochs 1
```

### Evaluation

```bash
python -m solar_knowledge.eval_full --weights models/latest.h5 --split 2023-05:2025-04
```

## Data Sources

- SDO/HMI SHARP data: https://jsoc1.stanford.edu/data/hmi/fits/
- NASA SDO: https://sdo.gsfc.nasa.gov/data/dataaccess.php

## Visualisation Resources

- https://spacebook.com/explorer
- https://cesium.com/blog/2024/11/26/spacebook-makes-space-data-accessible-for-all-with-cesiumjs/

# Key Files

- `models/SolarKnowledge_model.py` - The main solar flare prediction model
- `models/SolarKnowledge_run_all_trainings.py` - Script to train models
- `models/SolarKnowledge_run_all_tests.py` - Script to test models
- `scripts/model_tracker.py` - Track and compare model performance
- `docs/README_model_tracker.md` - Documentation for model tracking

# Usage

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
