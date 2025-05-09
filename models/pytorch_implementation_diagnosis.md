# PyTorch Implementation Diagnosis

## Problem

The PyTorch implementation of SolarKnowledge shows:

- Poor test performance (TSS near 0)
- Extremely skewed predictions (majority one class)
- Large discrepancy compared to TensorFlow implementation

## Findings

1. **Probability Distribution**: The raw model predictions have a mean positive probability of 0.2241, heavily skewed toward negative class.

2. **Label Distributions**: 
   - Training: 54.01% positive
   - Testing: 67.85% positive

3. **After Calibration**: 
   - Mean calibrated probability: 0.6786
   - Best threshold: 0.6831
   - TSS improvement: from -0.001 to 0.0672
   - More balanced predictions: 46.99% positive vs 0.36% with default threshold

## Root Causes

The PyTorch model has difficulty generalizing to the test data despite having the same architecture as the TensorFlow model. Potential issues:

1. **Initialization Differences**: Different initializers between PyTorch and TensorFlow
2. **Training Procedure**: Differences in optimization, learning rate, or regularization
3. **Data Handling**: Feature normalization inconsistencies between train and test
4. **Loss Implementation**: The focal loss implementation might not match TensorFlow's exactly
5. **Distribution Shift**: The PyTorch model might be more sensitive to train/test distribution shift

## Recommendations

1. **Initialization**: Ensure PyTorch and TensorFlow use exactly the same initialization
   - Print weight histograms from both models
   - Match initialization schemes precisely

2. **Optimization**:
   - Verify Adam parameters match (epsilon, beta1, beta2)
   - Check gradient clipping behavior

3. **Data Processing**:
   - Ensure features are normalized consistently across train/test
   - Apply same preprocessing steps

4. **Regularization**:
   - Add more dropout between transformer blocks
   - Try weight decay to match TF L2 regularization
   - Use more aggressive early stopping

5. **Ensemble Approaches**:
   - Consider multiple PyTorch models with different random seeds
   - Use model calibration as part of the pipeline

6. **Dataset Handling**:
   - Review train/test split methodology
   - Check for temporal correlations in the data

7. **Training Procedure**:
   - Try larger batch sizes or accumulate gradients
   - Experiment with different learning rates and schedules

The most promising approach is to integrate calibration directly into the model pipeline and enforce more consistent normalization between train and test data. 