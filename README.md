# EVEREST: Solar Flare Prediction System

![CI](https://github.com/antanaszilinskas/masters-project/actions/workflows/ci.yml/badge.svg?branch=pytorch-rewrite)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15570398.svg)](https://doi.org/10.5281/zenodo.15570398)

**EVEREST** (An Extreme-Value/Evidential-Retentive Transformer for Operational Solar-Flare Forecasting) is a state-of-the-art solar flare prediction system using transformer-based architectures trained on SDO/HMI SHARP magnetogram data.

## 🌟 Key Features

- **Multi-Framework Support**: Both TensorFlow and PyTorch implementations
- **Comprehensive Metrics**: 15+ evaluation metrics including space weather standard TSS and BSS
- **Confidence Intervals**: Bootstrap statistical analysis for robust evaluation
- **Multiple Prediction Horizons**: 24h, 48h, and 72h forecast windows
- **Flare Class Prediction**: C, M, M5, and X-class solar flare prediction
- **Energy Efficient**: Built-in carbon emissions tracking with CodeCarbon

## 📂 Repository Structure

This repository serves dual purposes: **reproducibility** and **research showcase**. The `reproducibility/` folder provides a complete, self-contained pipeline for assessors to reproduce all results from scratch, while the remaining directories demonstrate the full scope of research work, including experimental implementations, ablation studies, and comprehensive evaluation frameworks.

```
masters-project/
├── reproducibility/                 # Complete reproduction pipeline
│   ├── data/                        # Data download and preprocessing
│   ├── training/                    # Model training scripts
│   ├── evaluation/                  # Testing and evaluation
│   ├── environment/                 # Environment setup
│   └── README.md                    # Step-by-step reproduction guide
├── models/                          # Main TensorFlow implementation
│   ├── SolarKnowledge_model.py      # Core TensorFlow model
│   ├── SolarKnowledge_run_all_trainings.py
│   ├── SolarKnowledge_run_all_tests.py
│   ├── SolarKnowledge_model_pytorch.py  # PyTorch implementation
│   ├── hpo/                         # Hyperparameter optimization
│   ├── ablation/                    # Ablation studies
│   └── training/                    # Training configurations
├── nature_models/                   # Enhanced evaluation models
│   ├── SolarKnowledge_model.py      # Enhanced TensorFlow model
│   ├── SolarKnowledge_run_all_tests.py  # Comprehensive metrics
│   └── README_enhanced_metrics.md   # Metrics documentation
├── tests/                           # Unit tests and CI tests
├── config/                          # Configuration files
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment
└── Nature_data/                     # Training data (local)
```

## 🔄 Reproducibility

**For assessors and researchers seeking to reproduce results:** Navigate to the `reproducibility/` folder which contains a complete, self-contained pipeline including:

- **Data acquisition**: Automated scripts for downloading and preprocessing SDO/HMI SHARP data
- **Environment setup**: Containerized environments ensuring consistent dependencies
- **Model training**: Streamlined training scripts with optimal hyperparameters
- **Evaluation**: Comprehensive testing and metrics generation
- **Documentation**: Step-by-step reproduction guide

```bash
cd reproducibility/
# Follow the README.md for complete reproduction pipeline
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU training, optional)
- 16GB+ RAM recommended
- 50GB+ free disk space for full dataset

### Installation

#### Option 1: Conda (Recommended for GPU)

```bash
# Clone the repository
git clone https://github.com/antanaszilinskas/masters-project.git
cd masters-project

# Create and activate conda environment
conda env create -f environment.yml
conda activate sk-env

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### Option 2: pip (CPU or existing CUDA setup)

```bash
# Clone the repository
git clone https://github.com/antanaszilinskas/masters-project.git
cd masters-project

# Install dependencies
pip install -r requirements.txt

# For PyTorch branch specifically
git checkout pytorch-rewrite
pip install -r requirements.txt
```

### Quick Test

```bash
# Run the test suite to verify installation
python -m pytest tests/ -v

# Test model imports
python -c "from models.SolarKnowledge_model import SolarKnowledge; print('✓ TensorFlow model import successful')"
python -c "from models.SolarKnowledge_model_pytorch import SolarKnowledge; print('✓ PyTorch model import successful')"
```

## 🎯 Usage

### Training Models

#### TensorFlow (Main Branch)

```bash
# Train on all configurations (C, M, M5 classes × 24h, 48h, 72h windows)
cd models
python SolarKnowledge_run_all_trainings.py

# Train specific configuration
python SolarKnowledge_model.py --flare_class M --time_window 24 --epochs 100
```

#### PyTorch (pytorch-rewrite branch)

```bash
# Switch to PyTorch branch
git checkout pytorch-rewrite

# Train models
cd models
python SolarKnowledge_run_all_trainings_pytorch.py

# Train with specific parameters
python SolarKnowledge_model_pytorch.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Evaluation and Testing

#### Standard Evaluation

```bash
# Run comprehensive evaluation with enhanced metrics
cd nature_models
python SolarKnowledge_run_all_tests.py

# This generates:
# - Individual model performance reports
# - Comprehensive JSON results
# - Bootstrap confidence intervals
# - Summary tables for publication
```

#### PyTorch Evaluation

```bash
# PyTorch evaluation (pytorch-rewrite branch)
cd models
python SolarKnowledge_run_all_tests_pytorch.py
```

### Key Output Files

- **Comprehensive Results**: `nature_models/solarknowledge_comprehensive_results.json`
- **Training Logs**: `models/logs/`
- **Model Weights**: `models/weights/` or `nature_models/models/`
- **Performance Plots**: `models/*.png`

## 📊 Understanding Results

### Key Metrics

- **TSS (True Skill Statistic)**: Primary space weather metric (-1 to 1, >0.5 excellent)
- **Precision**: Fraction of predicted flares that occurred
- **Recall**: Fraction of actual flares correctly predicted  
- **Brier Score**: Probabilistic accuracy (lower better, <0.1 excellent)
- **ECE**: Calibration quality (lower better, <0.05 well-calibrated)

### Sample Results Interpretation

```json
{
  "24": {
    "M": {
      "TSS": 0.713,           // Excellent skill
      "precision": 0.845,     // 84.5% of predictions correct
      "recall": 0.723,        // Caught 72.3% of actual flares
      "Brier_Score": 0.089,   // Well-calibrated probabilities
      "ECE": 0.039           // Excellent calibration
    }
  }
}
```

## 🔬 Research Features

### Hyperparameter Optimization

```bash
cd models/hpo
python hpo_run.py --n_trials 100 --flare_class M --time_window 24
```

### Ablation Studies

```bash
cd models/ablation
python ablation_study.py --components "focal_loss,attention,dropout"
```

### Model Tracking and Comparison

```bash
cd models
python model_tracking.py scan
python model_tracking.py compare --metric TSS --top 5
```

## 📈 Continuous Integration

The project uses GitHub Actions for automated testing:

- **Unit Tests**: `pytest tests/`
- **Model Import Tests**: Verify all model imports work
- **Syntax Validation**: Check Python syntax across codebase
- **Energy Tracking**: Monitor computational carbon footprint
- **Multi-Framework Testing**: Test both TensorFlow and PyTorch branches

## 📚 Data Sources

- **SDO/HMI SHARP**: Space-weather HMI Active Region Patches
- **NOAA GOES**: X-ray flare event catalog
- **Processed Dataset**: Available at [DOI:10.5281/zenodo.15570398](https://doi.org/10.5281/zenodo.15570398)

### Data Pipeline

The full data processing pipeline is available in `archive/data/data_pipeline/`:

1. Download raw SHARP data from JSOC
2. Clean and merge SHARP parameters  
3. Add flare labels from NOAA catalog
4. Engineer temporal and statistical features
5. Create chronological train/validation/test splits

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_full.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/ -v
```

### Branch Strategy

- **main**: Stable TensorFlow implementation
- **pytorch-rewrite**: PyTorch implementation  
- **develop**: Development and experimental features

## 📖 Documentation

- **Enhanced Metrics**: `nature_models/README_enhanced_metrics.md`
- **Training Guide**: `models/TRAINING_METRICS_GUIDE.md`
- **HPO Setup**: `models/hpo/CLUSTER_SETUP_GUIDE.md`
- **Production Setup**: `docs/PRODUCTION_TRAINING_SETUP.md`

## 🎓 Citation

If you use EVEREST in your research, please cite:

```bibtex
@misc{zilinskas2025everest,
  author = {Zilinskas, Antanas},
  title = {EVEREST: Enhanced Solar Flare Prediction using Transformers},
  year = {2025},
  url = {https://github.com/antanaszilinskas/masters-project},
  doi = {10.5281/zenodo.15570398}
}
```

## 📄 License

This project is licensed under the CC BY 4.0 License - see the LICENSE file for details.

Raw solar data from NASA/SDO and NOAA are in the public domain.

---

**⚡ Getting Started Checklist:**

**For Full Reproduction:**
- [ ] Navigate to `reproducibility/` folder
- [ ] Follow the step-by-step reproduction guide
- [ ] Execute complete pipeline from data download to final results

**For Research and Development:**
- [ ] Clone repository and install dependencies
- [ ] Run `pytest tests/` to verify setup
- [ ] Download data or use existing `Nature_data/`
- [ ] Try training: `cd models && python SolarKnowledge_run_all_trainings.py`
- [ ] Run evaluation: `cd nature_models && python SolarKnowledge_run_all_tests.py`
- [ ] Check results in generated JSON files

For issues or questions, please open a GitHub issue or refer to the documentation in the respective directories.
