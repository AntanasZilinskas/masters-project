import argparse
import os
import sys
from typing import Dict
import warnings
import csv
from datetime import datetime

import torch
from ptflops import get_model_complexity_info

# TensorFlow imports for the other models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from models.solarknowledge_ret_plus import RETPlusModel
from models.SolarKnowledge_model import SolarKnowledge

# Fix for old TensorFlow syntax in SolarFlareNet
if not hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior = lambda: None

sys.path.append(os.path.join(repo_root, 'archive', 'nature_models'))
from SolarFlareNet_model import SolarFlareNet

# Group definitions aligned with the thesis table
GROUPS: Dict[str, list] = {
    "Embedding + PE": ["embedding", "norm", "pos"],
    "Transformer ×6": ["transformers"],
    "Attention bottleneck": ["att_pool"],
    "Classification head": ["head", "logits"],
    "Evidential head": ["nig"],
    "EVT head": ["gpd"],
    "Precursor head": ["precursor_head"],
}


def format_count(count: int, unit: str = "") -> str:
    """Format parameter/FLOP counts with adaptive units and appropriate precision."""
    if count >= 1e9:
        return f"{count/1e9:8.3f} G{unit}"
    elif count >= 1e6:
        return f"{count/1e6:8.3f} M{unit}"
    elif count >= 1e3:
        return f"{count/1e3:8.3f} k{unit}"
    else:
        return f"{count:8.0f}  {unit}"


def estimate_memory_usage(params: int, flops: int, batch_size: int, precision: str = "fp32") -> int:
    """Estimate peak GPU memory usage in MB."""
    bytes_per_param = 4 if precision == "fp32" else 2  # fp16
    
    # Model weights
    model_memory = params * bytes_per_param
    
    # Activation memory (rough estimate: ~3x parameters for gradients + activations)
    activation_memory = params * bytes_per_param * 3 * batch_size
    
    # Add some overhead for temporary buffers
    overhead = model_memory * 0.2
    
    total_bytes = model_memory + activation_memory + overhead
    return int(total_bytes / (1024 * 1024))  # Convert to MB


def calculate_efficiency_metrics(params: int, flops: int, accuracy: float = None) -> Dict[str, float]:
    """Calculate various efficiency metrics."""
    metrics = {
        "params_per_gflop": params / (flops / 1e9) if flops > 0 else 0,
        "flops_per_param": flops / params if params > 0 else 0,
    }
    
    if accuracy is not None:
        metrics["accuracy_per_mparam"] = accuracy / (params / 1e6) if params > 0 else 0
        metrics["accuracy_per_gflop"] = accuracy / (flops / 1e9) if flops > 0 else 0
    
    return metrics


def get_tf_model_flops(model, input_shape):
    """Estimate FLOPs for TensorFlow model using tf.profiler."""
    try:
        # Create a concrete function
        @tf.function
        def model_fn(x):
            return model(x, training=False)
        
        # Create dummy input
        dummy_input = tf.random.normal((1,) + input_shape)
        concrete_func = model_fn.get_concrete_function(dummy_input)
        
        # Run profiler
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        
        opts = ProfileOptionBuilder.float_operation()
        flops = profile(concrete_func.graph, options=opts)
        
        if flops is None:
            return 0
        return flops.total_float_ops
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs for TensorFlow model: {e}")
        # Fallback: rough estimation based on parameters
        return model.count_params() * 2


def analyze_everest_model(T: int, F: int, batch: int):
    """Analyze EVEREST (RETPlusModel) complexity."""
    model = RETPlusModel((T, F))

    # Get total stats from ptflops
    macs_total, params_total = get_model_complexity_info(
        model,
        (T, F),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )

    # Count parameters by walking through the model structure
    params_by_group = {group: 0 for group in GROUPS}
    total_counted = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        param_count = param.numel()
        total_counted += param_count
        
        # Assign to groups based on parameter name
        matched = False
        for group, prefixes in GROUPS.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                params_by_group[group] += param_count
                matched = True
                break
        
        if not matched:
            print(f"Warning: Unmatched parameter '{name}' with {param_count} params")

    # Estimate FLOPs by scaling ptflops total based on parameter ratios
    macs_by_group = {}
    for group in GROUPS:
        if total_counted > 0:
            param_ratio = params_by_group[group] / total_counted
            macs_by_group[group] = int(macs_total * param_ratio)
        else:
            macs_by_group[group] = 0

    total_flops = 2 * macs_total * batch
    return params_total, total_flops, params_by_group, {k: 2 * v * batch for k, v in macs_by_group.items()}


def analyze_solarknowledge_model(T: int, F: int, batch: int):
    """Analyze SolarKnowledge model complexity."""
    # Build SolarKnowledge model
    sk_model = SolarKnowledge()
    model = sk_model.build_base_model(
        input_shape=(T, F),
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=6,
        dropout_rate=0.2,
        num_classes=2
    )
    
    params_total = model.count_params()
    flops_total = get_tf_model_flops(model, (T, F)) * batch
    
    return params_total, flops_total


def analyze_solarflarenet_model(T: int, F: int, batch: int):
    """Analyze SolarFlareNet model complexity."""
    # Build SolarFlareNet model
    sfn_model = SolarFlareNet()
    model = sfn_model.build_base_model(
        input_shape=(T, F),
        dropout=0.4,
        b=4,  # 4 transformer blocks
        nclass=2,
        verbose=False
    )
    sfn_model.models()  # Finalize the model
    
    params_total = sfn_model.model.count_params()
    flops_total = get_tf_model_flops(sfn_model.model, (T, F)) * batch
    
    return params_total, flops_total


def save_results_to_csv(models_data: Dict, T: int, F: int, batch: int, output_file: str = None):
    """Save analysis results to CSV file for thesis documentation."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"complexity_analysis_{timestamp}.csv"
    
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", output_file)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Input_Config', 'Parameters', 'FLOPs', 'Memory_MB_fp32', 'Memory_MB_fp16', 
                     'Params_per_GFLOP', 'FLOPs_per_Param', 'Architecture_Type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for model_name, data in models_data.items():
            params = data['params']
            flops = data['flops']
            memory_fp32 = estimate_memory_usage(params, flops, batch, "fp32")
            memory_fp16 = estimate_memory_usage(params, flops, batch, "fp16")
            efficiency = calculate_efficiency_metrics(params, flops)
            
            # Determine architecture type
            if model_name == "EVEREST":
                arch_type = "Transformer + Evidential + EVT"
            elif model_name == "SolarKnowledge":
                arch_type = "Transformer"
            else:  # SolarFlareNet
                arch_type = "CNN + LSTM + Transformer"
            
            writer.writerow({
                'Model': model_name,
                'Input_Config': f"T={T}_F={F}_B={batch}",
                'Parameters': params,
                'FLOPs': flops,
                'Memory_MB_fp32': memory_fp32,
                'Memory_MB_fp16': memory_fp16,
                'Params_per_GFLOP': round(efficiency['params_per_gflop'], 2),
                'FLOPs_per_Param': round(efficiency['flops_per_param'], 2),
                'Architecture_Type': arch_type
            })
    
    print(f"\n📊 Results saved to: {output_path}")
    return output_path


def print_efficiency_analysis(models_data: Dict, batch: int):
    """Print additional efficiency and scaling analysis."""
    print("\nEFFICIENCY ANALYSIS")
    print("-" * 60)
    print(f"{'Model':<15s} {'Mem(MB)':<10s} {'Params/GFLOP':<12s} {'FLOPs/Param':<12s}")
    print("-" * 60)
    
    for model_name, data in models_data.items():
        params = data['params']
        flops = data['flops']
        memory = estimate_memory_usage(params, flops, batch)
        efficiency = calculate_efficiency_metrics(params, flops)
        
        print(f"{model_name:<15s} {memory:<10d} {efficiency['params_per_gflop']:<12.1f} {efficiency['flops_per_param']:<12.1f}")
    
    print("-" * 60)
    print("\nKEY INSIGHTS:")
    print("• Lower Params/GFLOP = more compute-intensive")
    print("• Higher FLOPs/Param = more operations per parameter")
    print("• Memory estimates include model + activations + gradients")


def main(T: int, F: int, batch: int, save_csv: bool = False):
    print("=" * 80)
    print("MODEL COMPLEXITY COMPARISON")
    print("=" * 80)
    print(f"Input configuration: T={T}, F={F}, batch={batch}")
    print()

    # Analyze all three models
    models_data = {}
    
    # 1. EVEREST (RETPlusModel)
    print("Analyzing EVEREST (RETPlusModel)...")
    everest_params, everest_flops, everest_groups, everest_flops_groups = analyze_everest_model(T, F, batch)
    models_data['EVEREST'] = {
        'params': everest_params,
        'flops': everest_flops,
        'groups': everest_groups,
        'flops_groups': everest_flops_groups
    }
    
    # 2. SolarKnowledge
    print("Analyzing SolarKnowledge...")
    sk_params, sk_flops = analyze_solarknowledge_model(T, F, batch)
    models_data['SolarKnowledge'] = {
        'params': sk_params,
        'flops': sk_flops
    }
    
    # 3. SolarFlareNet
    print("Analyzing SolarFlareNet...")
    sfn_params, sfn_flops = analyze_solarflarenet_model(T, F, batch)
    models_data['SolarFlareNet'] = {
        'params': sfn_params,
        'flops': sfn_flops
    }
    
    print()
    
    # Display overall comparison
    print("OVERALL MODEL COMPARISON")
    print("-" * 60)
    print(f"{'Model':<20s} {'Params':>15s} {'FLOPs':>20s}")
    print("-" * 60)
    
    for model_name, data in models_data.items():
        params_str = format_count(data['params'])
        flops_str = format_count(data['flops'])
        print(f"{model_name:<20s} {params_str:>15s} {flops_str:>20s}")
    
    print("-" * 60)
    print()
    
    # Display detailed EVEREST breakdown
    print("EVEREST (RETPlusModel) COMPONENT BREAKDOWN")
    print("-" * 70)
    print(f"{'Module':<25s} {'Params':>12s} {'FLOPs':>15s}")
    print("-" * 70)

    total_p = total_f = 0
    for group in GROUPS:
        p = models_data['EVEREST']['groups'][group]
        f = models_data['EVEREST']['flops_groups'][group]
        total_p += p
        total_f += f
        
        params_str = format_count(p)
        flops_str = format_count(f)
        print(f"{group:<25s} {params_str:>12s} {flops_str:>15s}")

    print("-" * 70)
    total_params_str = format_count(total_p)
    total_flops_str = format_count(total_f)
    print(f"{'Total':<25s} {total_params_str:>12s} {total_flops_str:>15s}")
    
    # Add efficiency analysis
    print_efficiency_analysis(models_data, batch)
    
    # Save results if requested
    if save_csv:
        save_results_to_csv(models_data, T, F, batch)
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute complexity for all solar flare prediction models.")
    parser.add_argument("--timesteps", "-T", type=int, default=10, help="Sequence length")
    parser.add_argument("--features", "-F", type=int, default=9, help="Number of input features")
    parser.add_argument("--batch", "-B", type=int, default=1, help="Batch size for FLOP scaling")
    parser.add_argument("--save-csv", action="store_true", help="Save results to CSV file")
    args = parser.parse_args()
    main(args.timesteps, args.features, args.batch, args.save_csv)