# EVEREST Component Ablation Study

## Focused Study: Component Ablations Only

Based on your request to omit time series duration analysis, this study focuses exclusively on **component ablations** to understand the contribution of each architectural component.

### ✅ **Component Ablation Study (35 experiments)**
- **Component ablations**: 7 variants × 5 seeds = 35 experiments
- **Focus**: Understanding architectural component contributions
- **Manageable scope**: Faster execution and analysis

## Component Ablation Breakdown

### **Component Ablations (35 experiments)**

Tests the contribution of each model component by removing or modifying it:

| Variant | Description | Components Modified |
|---------|-------------|-------------------|
| `full_model` | Complete EVEREST model | Baseline (all components) |
| `no_evidential` | Remove evidential uncertainty | `use_evidential=False` |
| `no_evt` | Remove EVT tail modeling | `use_evt=False` |
| `mean_pool` | Replace attention with mean pooling | `use_attention_bottleneck=False` |
| `cross_entropy` | Use only focal loss | `use_evidential=False, use_evt=False` |
| `no_precursor` | Remove precursor head | `use_precursor=False` |
| `fp32_training` | Full precision training | Same as full_model but FP32 |

**Seeds**: 0, 1, 2, 3, 4 (5 repetitions per variant)

## Technical Implementation

### Experiment Mapping

**Jobs 1-35**: Component ablations
- Job 1-5: `full_model` seeds 0-4
- Job 6-10: `no_evidential` seeds 0-4
- Job 11-15: `no_evt` seeds 0-4
- Job 16-20: `mean_pool` seeds 0-4
- Job 21-25: `cross_entropy` seeds 0-4
- Job 26-30: `no_precursor` seeds 0-4
- Job 31-35: `fp32_training` seeds 0-4

### Fixed Configuration
- **Input shape**: (10, 9) - standard 10 timestep sequence
- **Target**: M5-class, 72h window (optimal from HPO)
- **Hyperparameters**: Optimal values from HPO study

## Usage

### Test Component Study
```bash
cd models/ablation
qsub cluster/test_component_ablation.pbs
```

### Run Component Study
```bash
cd models/ablation
qsub cluster/submit_component_ablation.pbs
```

### Monitor Progress
```bash
# Check job status
qstat -u $USER

# Count running/completed jobs
qstat -u $USER | grep everest_component_ablation | wc -l

# Check specific job output
tail -f everest_component_ablation.o12345
```

## Expected Results

After completion, you'll have:

1. **35 trained models** - one for each component variant and seed
2. **Performance metrics** for each variant and seed
3. **Statistical significance** from 5 repetitions per variant
4. **Component importance** ranking from ablations

## Scientific Value

### Component Ablations Answer:
- Which components contribute most to performance?
- Is evidential uncertainty worth the complexity?
- Does EVT tail modeling improve rare event detection?
- How much does attention pooling help vs. mean pooling?
- What's the impact of the precursor head?
- Does mixed precision training affect results?

## Benefits of Focused Approach

### ✅ **Advantages**
- **Faster execution**: 35 vs 60 experiments (42% reduction)
- **Clearer focus**: Component analysis without sequence complexity
- **Easier analysis**: Single dimension of variation
- **Sufficient insights**: Core architectural questions answered

### 📊 **Still Comprehensive**
- **7 key variants** covering all major components
- **5 seeds** for statistical significance
- **Optimal hyperparameters** from HPO study
- **Same rigorous methodology** as working HPO

## Files Created

1. **`run_ablation_exact_hpo.py`** - Component ablation runner (simplified)
2. **`cluster/submit_component_ablation.pbs`** - 35-job array submission script
3. **`cluster/test_component_ablation.pbs`** - Component-focused test script
4. **`COMPONENT_ABLATION_STUDY.md`** - This focused documentation

## Why This Approach Makes Sense

1. **Component analysis is the core question** - understanding which architectural pieces matter most
2. **Sequence length can be studied separately** - if needed later, it's a simpler follow-up study
3. **Faster iteration** - get component insights quickly, then decide on sequence analysis
4. **Resource efficient** - 35 experiments vs 60 saves significant cluster time

The implementation provides a **focused, scientifically rigorous component ablation study** that will give you clear insights into your EVEREST model's architectural components and their relative importance. 