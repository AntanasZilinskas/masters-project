# EVEREST Ablation Study - EXACT HPO Pattern Solution

## Problem Analysis: Key Differences Between HPO and Ablation

After detailed comparison, several **critical differences** were found between your working HPO setup and the failing ablation approach:

### 1. **Data Loading Pattern** ❌
**HPO (Working)**: Loads data once in `HPOObjective.__init__()` and caches it
```python
def _load_data(self) -> None:
    self.X_train, self.y_train = get_training_data(self.time_window, self.flare_class)
    self.X_val, self.y_val = get_testing_data(self.time_window, self.flare_class)
    # Converts to numpy and caches for all trials
```

**Ablation (Failing)**: Loads data every time in `train_ablation_variant()`
```python
# This happens for EVERY experiment (60 times!)
X_train, y_train = get_training_data("72", "M5")
X_test, y_test = get_testing_data("72", "M5")
```

### 2. **Model Creation Pattern** ❌
**HPO (Working)**: Uses `RETPlusWrapper` constructor directly
```python
wrapper = RETPlusWrapper(
    input_shape=model_config["input_shape"],
    use_evidential=model_config["use_evidential"],
    # ... other flags
)
# Model is created inside the wrapper with proper initialization
```

**Ablation (Failing)**: Creates model separately then replaces it
```python
wrapper = RETPlusWrapper(...)  # Creates one model
ablation_model = RETPlusModel(...)  # Creates another model  
wrapper.model = ablation_model  # REPLACES the model!
```

**This is dangerous because:**
- The wrapper's internal state might not match the new model
- Optimizer might be pointing to the wrong parameters
- Device placement might be inconsistent

### 3. **Training Method** ❌
**HPO (Working)**: Uses `RETPlusWrapper.train()` method
```python
# Uses the wrapper's built-in training method with proper validation
metrics = self._train_and_evaluate(model, hyperparams, epochs, trial)
```

**Ablation (Failing)**: Uses custom `_train_with_validation()` method
```python
# Custom training loop that might have different behavior
results = self._train_with_validation(model, X_train, y_train, ...)
```

### 4. **Environment Setup** ❌
**HPO (Working)**: Comprehensive environment setup and validation
```python
def _setup_reproducibility(self) -> None:
    seed = REPRODUCIBILITY_CONFIG["random_seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ... comprehensive setup with proper error handling
```

**Ablation (Failing)**: Basic validation but missing key setup steps

## Solution: EXACT HPO Pattern

Created `run_ablation_exact_hpo.py` that follows the **exact same pattern** as your working HPO:

### ✅ **Key Features**

1. **Identical Data Loading**
   - Loads data once in `__init__()` and caches it
   - Uses exact same error handling as HPO
   - Converts to numpy arrays with same pattern

2. **Identical Model Creation**
   - Uses `RETPlusWrapper` constructor directly
   - Lets wrapper create the model internally
   - No dangerous model replacement

3. **Identical Training Method**
   - Uses `wrapper.train()` method (same as HPO)
   - Same early stopping and validation logic
   - Same loss weight scheduling

4. **Identical Environment Setup**
   - Same conda activation pattern
   - Same GPU validation
   - Same reproducibility setup

### ✅ **Files Created**

1. **`run_ablation_exact_hpo.py`** - Complete ablation runner (component + sequence) following exact HPO pattern
2. **`cluster/submit_complete_ablation.pbs`** - Complete cluster submission for all 60 experiments
3. **`cluster/test_complete_ablation.pbs`** - Test script to verify both component and sequence ablations work

## Usage

### Test the Complete Study
```bash
cd models/ablation
qsub cluster/test_complete_ablation.pbs
```

### Run Complete Study (60 experiments)
```bash
cd models/ablation  
qsub cluster/submit_complete_ablation.pbs
```

## Expected Results

This approach should work because it:

1. **Uses your exact working environment** - same conda activation, same modules
2. **Uses your exact working data loading** - same functions, same caching pattern  
3. **Uses your exact working model creation** - same wrapper initialization
4. **Uses your exact working training method** - same `wrapper.train()` call
5. **Uses your exact working validation** - same GPU checks, same error handling

## Complete Experiment Configuration

**Total: 60 experiments**

### Component Ablations (35 experiments)
- **7 variants × 5 seeds = 35 experiments**
- **Variants**: full_model, no_evidential, no_evt, mean_pool, cross_entropy, no_precursor, fp32_training
- **Seeds**: 0, 1, 2, 3, 4

### Sequence Length Ablations (25 experiments)  
- **5 variants × 5 seeds = 25 experiments**
- **Sequence variants**: seq_5, seq_7, seq_10, seq_15, seq_20
- **Seeds**: 0, 1, 2, 3, 4

### Configuration
- **Target**: M5-class, 72h window (same as HPO optimal target)
- **Hyperparameters**: Optimal values from HPO study
- **Input shapes**: (5,9), (7,9), (10,9), (15,9), (20,9) for sequence variants

## Monitoring

Check test success with:
```bash
# Check job status
qstat -u $USER

# View output
tail -f test_hpo_pattern.o*

# Look for success indicators
grep -E "(✅|🏁|completed)" test_hpo_pattern.o*
```

## Why This Should Work

The key insight is that your HPO study **already works perfectly** on the cluster. By following the **exact same pattern**, we eliminate all the variables that could cause failures:

- ✅ Same environment setup
- ✅ Same data loading approach  
- ✅ Same model creation pattern
- ✅ Same training methodology
- ✅ Same validation logic

This is essentially "copying" your working HPO approach and adapting it for ablation studies, rather than creating a new approach that might have subtle differences causing cluster failures. 