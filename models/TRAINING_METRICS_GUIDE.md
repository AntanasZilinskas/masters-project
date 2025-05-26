# EVEREST Training Metrics & Environmental Impact Tracking

## Overview
The EVEREST model training now includes comprehensive metrics tracking including timing, GPU monitoring, and CO2 emissions measurement.

## Features Added

### ⏱️ **Training Timing**
- **Total training time** (seconds and hours)
- **Per-epoch timing** with statistics:
  - Average epoch time
  - Fastest/slowest epoch times
  - Complete epoch time series
- **Early stopping detection**

### 🖥️ **Hardware Information**
- **GPU type and memory** (e.g., "NVIDIA A100 80GB")
- **Mixed precision usage** detection
- **Device information** (CUDA/MPS/CPU)

### ⚡ **Power Monitoring** 
- **Real-time GPU power consumption** via `nvidia-smi`
- **60-second sampling intervals**
- **Power statistics**:
  - Average, min, max power draw (Watts)
  - Total monitoring duration
  - Number of power readings

### 🌱 **CO2 Emissions Tracking**
- **CodeCarbon v2.3** integration
- **Machine-level tracking** (CPU + GPU)
- **Automatic emissions calculation** in kg CO2eq
- **Project-specific tracking** per flare class and time window

## Installation

Install the required dependency:
```bash
pip install codecarbon==2.3.4
```

## Usage

The metrics are automatically collected during training:

```python
from solarknowledge_ret_plus import RETPlusWrapper

model = RETPlusWrapper((10, 9))
model_dir = model.train(X_train, y_train, flare_class="M", time_window="24")
```

## Output Example

During training, you'll see enhanced logging:
```
Training on: NVIDIA L40S (48 GB)
🌱 CO2 emissions tracking started with codecarbon

Epoch 1/100 - loss: 0.6234 - acc: 0.7123 - tss: 0.2456 - gamma: 0.00 - time: 45.2s
Epoch 2/100 - loss: 0.5987 - acc: 0.7234 - tss: 0.2567 - gamma: 0.04 - time: 43.8s
...

📊 Training completed in 2847.3s (0.79h)
   • Average epoch time: 42.1s
   • GPU: NVIDIA L40S
   • Average GPU power: 287.4W
   • CO2 emissions: 0.000234 kg CO2eq
```

## Stored Metrics

All metrics are saved in the model metadata under `training_metrics`:

```json
{
  "total_training_time_s": 2847.3,
  "total_training_time_h": 0.79,
  "average_epoch_time_s": 42.1,
  "fastest_epoch_time_s": 38.7,
  "slowest_epoch_time_s": 47.3,
  "epoch_times": [45.2, 43.8, ...],
  "epochs_completed": 67,
  "early_stopped": true,
  "gpu_info": {
    "gpu_name": "NVIDIA L40S", 
    "gpu_memory_gb": 48
  },
  "gpu_power_stats": {
    "average_power_w": 287.4,
    "max_power_w": 312.1,
    "min_power_w": 245.7,
    "power_readings": 48,
    "monitoring_duration_s": 2840.0
  },
  "co2_emissions_kg": 0.000234,
  "training_samples": 406584,
  "batch_size": 512,
  "mixed_precision": true
}
```

## Files Created

### Training Artifacts
- **Model weights**: `models/EVEREST-vX.Y.Z-CLASS-Th/model_weights.pt`
- **Metadata**: `models/EVEREST-vX.Y.Z-CLASS-Th/metadata.json`
- **Training history**: Loss, accuracy, TSS curves

### Environmental Tracking
- **CodeCarbon logs**: `models/hpo/emissions/`
- **Power monitoring**: Integrated in metadata
- **Project tracking**: Per-configuration emissions

## HPO Integration

During hyperparameter optimization, metrics are collected for **every trial**:
- Individual trial timing and efficiency
- Aggregate emissions across all trials  
- GPU utilization patterns
- Energy consumption per configuration

## Cluster Usage

On Imperial RCS cluster:
```bash
# Install codecarbon in your environment
conda activate everest_env
pip install codecarbon==2.3.4

# Submit HPO jobs (metrics automatically collected)
qsub models/hpo/cluster/run_hpo_array.pbs
```

## Benefits

1. **Performance optimization**: Identify bottlenecks and efficiency patterns
2. **Environmental impact**: Track and minimize carbon footprint
3. **Resource planning**: GPU utilization and power consumption insights
4. **Reproducibility**: Complete training environment documentation
5. **Cost analysis**: Time and energy consumption for budget planning 