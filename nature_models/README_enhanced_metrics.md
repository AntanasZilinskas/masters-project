# Enhanced SolarKnowledge Metrics for SolarFlareNet Comparison

## Overview

This document describes the comprehensive metrics evaluation implemented in `SolarKnowledge_run_all_tests.py` for direct comparison with the SolarFlareNet paper. The enhanced evaluation includes 15+ metrics organized into three tiers based on importance for space weather forecasting.

## Metrics Classification

### 🥇 TIER 1: Primary Metrics for SolarFlareNet Comparison

These are the most critical metrics for direct comparison with SolarFlareNet results:

#### 1. True Skill Statistic (TSS)
- **Formula**: TSS = Sensitivity + Specificity - 1
- **Range**: [-1, 1]
- **Interpretation**: 
  - TSS > 0: Skill above random chance
  - TSS = 0: No skill (random)
  - TSS < 0: Worse than random
- **Space Weather Context**: Primary metric for operational space weather evaluation
- **SolarFlareNet**: Key metric reported in their paper

#### 2. Precision
- **Formula**: Precision = TP / (TP + FP)
- **Range**: [0, 1]
- **Interpretation**: Fraction of predicted flares that actually occurred
- **Space Weather Context**: Critical for operational systems to minimize false alarms
- **SolarFlareNet**: Directly reported metric

#### 3. Recall (Sensitivity/POD)
- **Formula**: Recall = TP / (TP + FN)
- **Range**: [0, 1]
- **Interpretation**: Fraction of actual flares that were correctly predicted
- **Space Weather Context**: Critical for not missing dangerous events
- **SolarFlareNet**: Directly reported metric

#### 4. Balanced Accuracy (BACC)
- **Formula**: BACC = (Sensitivity + Specificity) / 2
- **Range**: [0, 1]
- **Interpretation**: Average of sensitivity and specificity, handles class imbalance
- **Space Weather Context**: More appropriate than raw accuracy for rare events
- **SolarFlareNet**: Reported metric

#### 5. Brier Score (BS)
- **Formula**: BS = (1/N) Σ(forecast_probability - actual_outcome)²
- **Range**: [0, 1]
- **Interpretation**: Lower is better; measures calibration and resolution
- **Space Weather Context**: Key probabilistic forecasting metric
- **SolarFlareNet**: Extensively reported with standard deviations

#### 6. Brier Skill Score (BSS)
- **Formula**: BSS = 1 - (BS_forecast / BS_reference)
- **Range**: [-∞, 1]
- **Interpretation**: 
  - BSS > 0: Better than climatology
  - BSS = 0: Same as climatology
  - BSS < 0: Worse than climatology
- **Reference**: Climatological forecast (BS_ref = event_rate × (1 - event_rate))
- **SolarFlareNet**: Reported metric

#### 7. Expected Calibration Error (ECE)
- **Formula**: ECE = Σ |accuracy_in_bin - confidence_in_bin| × proportion_in_bin
- **Range**: [0, 1]
- **Interpretation**: Lower is better; measures calibration quality
- **Space Weather Context**: Critical for operational decision-making
- **Operational Threshold**: Met Office uses < 0.04 as well-calibrated
- **SolarKnowledge Advantage**: Demonstrates Focal Loss calibration benefits

### 📈 TIER 2: Highly Recommended Metrics

#### 8. F1-Score
- **Formula**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: [0, 1]
- **Interpretation**: Harmonic mean of precision and recall
- **Context**: Good single metric balancing precision and recall

#### 9. ROC-AUC
- **Formula**: Area Under the Receiver Operating Characteristic Curve
- **Range**: [0, 1]
- **Interpretation**: Ability to discriminate between classes
- **Context**: Standard ML metric for binary classification

### 📋 TIER 3: Standard ML Metrics

#### 10. Accuracy
- **Formula**: Accuracy = (TP + TN) / (TP + TN + FP + FN)
- **Range**: [0, 1]
- **Interpretation**: Overall fraction of correct predictions
- **Note**: Less meaningful for imbalanced datasets like solar flares

## Confidence Intervals

### Bootstrap Method
- **Implementation**: 1000 bootstrap samples with replacement
- **Confidence Level**: 95% (2.5th and 97.5th percentiles)
- **Metrics Covered**: TSS, Precision, Recall, Balanced Accuracy, Brier Score, ROC-AUC
- **Purpose**: Quantify uncertainty and enable statistical comparison

## Key Comparison Tasks

Focus on these configurations for direct SolarFlareNet comparison:
- **C-24h**: C-class flares, 24-hour prediction window
- **M-24h**: M-class flares, 24-hour prediction window  
- **M5-24h**: M5+ flares, 24-hour prediction window
- **C-48h**: C-class flares, 48-hour prediction window

## Output Format

### JSON Structure
```json
{
  "metadata": {
    "generated_on": "ISO timestamp",
    "description": "Comprehensive evaluation description",
    "metrics_included": {
      "tier_1": ["TSS", "precision", "recall", ...],
      "tier_2": ["f1_score", "ROC_AUC"],
      "tier_3": ["accuracy"]
    },
    "comparison_target": "SolarFlareNet paper results",
    "key_tasks": ["C-24h", "M-24h", "M5-24h", "C-48h"]
  },
  "results": {
    "24": {
      "C": {
        "TSS": 0.713,
        "precision": 0.845,
        "recall": 0.723,
        "balanced_accuracy": 0.856,
        "Brier_Score": 0.089,
        "Brier_Skill_Score": 0.447,
        "ECE": 0.039,
        "TSS_CI": {"lower": 0.689, "upper": 0.737},
        "precision_CI": {"lower": 0.821, "upper": 0.869},
        ...
      }
    }
  }
}
```

### Summary Tables

#### TABLE 1: Primary Metrics
Shows all Tier 1 metrics in a compact format for quick comparison.

#### TABLE 2: Secondary Metrics  
Shows Tier 2 and 3 metrics plus detailed analysis metrics.

#### TABLE 3: Key Comparison Tasks
Highlights the four main tasks for SolarFlareNet comparison.

#### TABLE 4: Confidence Intervals
Shows 95% bootstrap confidence intervals for key metrics.

## Usage

```bash
cd nature_models
python SolarKnowledge_run_all_tests.py
```

### Expected Output
1. **Progress**: Real-time evaluation progress with emoji indicators
2. **Individual Results**: Detailed metrics for each configuration
3. **Summary Tables**: Four comprehensive summary tables
4. **JSON File**: `solarknowledge_comprehensive_results.json`
5. **Key Findings**: Interpretation guidelines for comparison

## Interpretation Guidelines

### TSS Interpretation
- **TSS > 0.5**: Excellent skill
- **TSS 0.3-0.5**: Good skill
- **TSS 0.1-0.3**: Marginal skill
- **TSS < 0.1**: Poor skill

### Brier Score Interpretation
- **BS < 0.1**: Excellent calibration
- **BS 0.1-0.2**: Good calibration
- **BS 0.2-0.3**: Moderate calibration
- **BS > 0.3**: Poor calibration

### ECE Interpretation
- **ECE < 0.05**: Well-calibrated (Met Office threshold: < 0.04)
- **ECE 0.05-0.1**: Moderately calibrated
- **ECE > 0.1**: Poorly calibrated

## Advantages for SolarFlareNet Comparison

1. **Direct Metric Compatibility**: All SolarFlareNet metrics included
2. **Statistical Rigor**: Bootstrap confidence intervals for uncertainty quantification
3. **Space Weather Focus**: TSS and BSS emphasize domain-specific evaluation
4. **Calibration Quality**: ECE demonstrates SolarKnowledge's probabilistic advantages
5. **Comprehensive Coverage**: 15+ metrics provide complete performance picture
6. **Automated Comparison**: Structured output ready for paper integration

## Technical Implementation

### Dependencies
- scikit-learn: Core ML metrics
- numpy: Numerical computations
- json: Output formatting
- datetime: Metadata timestamps

### Performance
- Bootstrap CI calculation: ~30 seconds per configuration
- Memory efficient: Processes one configuration at a time
- Robust error handling: Graceful degradation for missing models

### Extensibility
The modular design allows easy addition of new metrics:
1. Implement metric function
2. Add to `calculate_comprehensive_metrics()`
3. Update summary tables
4. Add to JSON metadata

## Future Enhancements

1. **Multi-seed Evaluation**: Support for multiple random seeds
2. **Statistical Tests**: Automated significance testing vs. SolarFlareNet
3. **Visualization**: Automated plots for paper figures
4. **Ensemble Metrics**: Support for model ensembles
5. **Cross-validation**: K-fold evaluation support 