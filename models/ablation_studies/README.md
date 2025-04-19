# Ablation Studies for SolarKnowledge Models

This directory contains systematic ablation studies to understand the impact of different model parameters and design choices on solar flare prediction performance.

## Purpose

Ablation studies involve systematically "removing" or modifying individual components of the model to measure their impact on performance. These studies help:

1. Identify the most important model components
2. Optimize hyperparameters based on empirical evidence
3. Understand sensitivity to different parameters
4. Guide future model development

## Available Studies

### 1. Sequence Length Study

Located in `sequence_length/`, this study examines how the length of input time sequences affects model performance. It tests various sequence lengths (ranging from 25 to 200 timesteps) and measures key metrics like accuracy, precision, recall, and TSS.

**Run the sequence length study:**

```bash
# For a specific flare class and time window
python ablation_sequence_length.py --flare-class M5 --time-window 24

# Or run for all flare classes and time windows
python ablation_sequence_length.py --all
```

Results are saved to the `sequence_length/results/` directory as JSON files and visualized as plots.

### 2. Future Studies

The following ablation studies are planned for future implementation:

- **Transformer Depth**: Testing different numbers of transformer blocks
- **Embedding Dimension**: Testing different embedding dimensions
- **Multi-Head Attention**: Testing different numbers of attention heads
- **Dropout Rate**: Testing different dropout rates
- **Learning Rate**: Testing different learning rate schedules

## Results Organization

Each ablation study has its own directory with:

- A Python script to run the study
- A `results/` subdirectory containing:
  - JSON files with detailed metrics for each tested configuration
  - Visualization plots showing performance trends
  - Summary reports

## Interpreting Results

When analyzing ablation study results, consider:

1. **Performance Trends**: Look for patterns in how metrics change as parameters vary
2. **Precision-Recall Tradeoffs**: Some configurations may improve precision at the cost of recall or vice versa
3. **Computation Requirements**: Some configurations may perform slightly better but require significantly more resources
4. **Class-Specific Effects**: Note that optimal configurations may differ between flare classes (C, M, M5)

## Adding New Studies

To add a new ablation study:

1. Create a new directory under `ablation_studies/`
2. Create a Python script following the pattern of existing studies
3. Include result saving, visualization, and documentation
4. Update this README to describe the new study

## Contact

For questions or suggestions regarding these ablation studies, please contact Antanas Zilinskas. 