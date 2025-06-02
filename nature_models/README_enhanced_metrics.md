# Enhanced SolarKnowledge Model Evaluation

## Overview

The enhanced `SolarKnowledge_run_all_tests.py` script now calculates a comprehensive set of performance metrics for evaluating solar flare prediction models across different time windows and flare classes.

## 📊 Metrics Calculated

### Core Classification Metrics
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Average of sensitivity and specificity

### Domain-Specific Metrics
- **TSS (True Skill Statistic)**: Sensitivity + Specificity - 1
  - Range: [-1, 1], with 1 being perfect and 0 being no skill
  - Preferred metric in space weather prediction

### Probabilistic Metrics
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Brier Score**: Mean squared difference between predicted probabilities and actual outcomes
  - Lower is better (0 = perfect, 1 = worst)
- **ECE (Expected Calibration Error)**: Measures how well predicted probabilities match actual frequencies
  - Lower is better (0 = perfect calibration)

### Confusion Matrix Statistics
- **True Positives/Negatives**: Correctly predicted classes
- **False Positives/Negatives**: Incorrectly predicted classes
- **Sensitivity**: True positive rate (recall for positive class)
- **Specificity**: True negative rate
- **Positive/Negative Predictive Value**: Precision for each class

## 🚀 Usage

```bash
cd nature_models
python SolarKnowledge_run_all_tests.py
```

## 📋 Output

### Console Output
- Real-time progress for each model configuration
- Detailed metrics for each time window × flare class combination
- Summary table with key metrics
- Classification reports with per-class statistics

### JSON Output
The script saves results to `solarknowledge_comprehensive_results.json`:

```json
{
    "24": {
        "C": {
            "accuracy": 0.8567,
            "TSS": 0.7134,
            "ROC_AUC": 0.9123,
            "Brier": 0.1234,
            "ECE": 0.0567,
            "f1_score": 0.7890,
            "precision": 0.8012,
            "recall": 0.7768,
            "true_positives": 156,
            "false_positives": 38,
            "true_negatives": 742,
            "false_negatives": 44,
            "status": "Success",
            "test_samples": 980,
            "positive_samples": 200,
            "negative_samples": 780
        }
    }
}
```

## 🔧 Model Configurations Tested

The script evaluates all combinations of:
- **Time Windows**: 24h, 48h, 72h
- **Flare Classes**: C, M, M5

Total: 9 model configurations

## 📊 Summary Table Format

```
Config          Accuracy   TSS     ROC-AUC  Brier   ECE     F1    
--------------------------------------------------------------------------------
C-24h           0.857      0.713   0.912    0.123   0.057   0.789
M-24h           0.823      0.646   0.887    0.145   0.078   0.712
M5-24h          0.901      0.802   0.945    0.089   0.034   0.845
...
```

## 🛠️ Error Handling

- **Missing Model Weights**: Gracefully handles missing model directories
- **Probability Calculation Errors**: Falls back to basic metrics if probabilistic metrics fail
- **Data Format Issues**: Robust handling of different prediction output formats
- **Edge Cases**: Zero division protection and empty data handling

## 📈 Metric Interpretation Guide

### TSS (True Skill Statistic)
- **> 0.8**: Excellent performance
- **0.6 - 0.8**: Good performance  
- **0.4 - 0.6**: Moderate performance
- **< 0.4**: Poor performance

### Brier Score
- **< 0.1**: Excellent calibration
- **0.1 - 0.2**: Good calibration
- **0.2 - 0.3**: Moderate calibration
- **> 0.3**: Poor calibration

### ECE (Expected Calibration Error)
- **< 0.05**: Well-calibrated
- **0.05 - 0.1**: Reasonably calibrated
- **0.1 - 0.2**: Moderately calibrated
- **> 0.2**: Poorly calibrated

## 🔄 Integration with CI/CD

The enhanced script integrates seamlessly with your existing CI/CD pipeline:

1. **Automated Testing**: Can be run as part of model validation
2. **Metric Tracking**: JSON output enables automated performance monitoring
3. **Regression Detection**: Compare metrics across model versions
4. **Publication Ready**: Formatted output suitable for research papers

## 📋 Dependencies

Required packages (already in your environment):
- `numpy`
- `scikit-learn` 
- `tensorflow`/`torch` (depending on model)
- `json` (built-in)

## 🎯 Next Steps

Consider adding:
- **Confidence Intervals**: Bootstrap sampling for robust metric estimates
- **Statistical Tests**: Compare model performance significance
- **Visualization**: ROC curves, calibration plots, reliability diagrams
- **Cross-Validation**: More robust performance estimates 