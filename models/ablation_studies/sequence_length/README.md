# Sequence Length Ablation Study

This study examines how varying the input sequence length affects the performance of SolarKnowledge models for solar flare prediction.

## Background

The length of the input sequence is a critical parameter for time-series models:
- **Too short**: The model may miss important temporal patterns
- **Too long**: The model may suffer from noise, computational inefficiency, or overfitting
- **Optimal length**: May vary depending on flare class and prediction window

This study systematically tests different sequence lengths to find the optimal configuration for each flare class and prediction window combination.

## Implementation Details

The study:
1. Takes the original input sequences and resamples them to different lengths using linear interpolation
2. Trains models with the same architecture on each resampled dataset
3. Measures key performance metrics for each sequence length
4. Visualizes the relationship between sequence length and performance

## Sequence Lengths Tested

The following sequence lengths are tested:
- 25 timesteps
- 50 timesteps
- 75 timesteps
- 100 timesteps
- 125 timesteps
- 150 timesteps
- 175 timesteps
- 200 timesteps

## Usage

To run the study for M5-class flares with a 24-hour prediction window:

```bash
python ablation_sequence_length.py --flare-class M5 --time-window 24
```

To run for all combinations of flare classes and prediction windows:

```bash
python ablation_sequence_length.py --all
```

## Results Interpretation

When analyzing the results, look for:

1. **Performance peaks**: There's often a "sweet spot" where performance metrics peak
2. **Precision vs. recall**: Sequence length may affect precision and recall differently
3. **Class differences**: Optimal sequence length may differ between C, M, and M5 flares
4. **Diminishing returns**: At some point, increasing sequence length may yield minimal gains

## Technical Details

The resampling technique used is linear interpolation, which preserves the shape of the time series while changing its resolution. The original features are maintained, just with a different temporal sampling.

For computational efficiency, the study uses:
- Fewer training epochs (50 instead of 100+)
- Early stopping with shorter patience
- The original hyperparameters otherwise

## Output Files

The study produces:
- **JSON result files**: Complete metrics for all tested sequence lengths
- **Plot files**: Visualizations showing how metrics change with sequence length
- **Console logs**: Detailed progress and results

All outputs are saved in the `results/` directory. 