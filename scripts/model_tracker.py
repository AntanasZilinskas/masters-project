import json
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import glob

MODELS_FILE = "weights/model_results.json"
WEIGHTS_BASE_DIR = "weights"


def setup_environment():
    """Create necessary files and directories if they don't exist."""
    if not os.path.exists(MODELS_FILE):
        # Ensure the weights directory exists
        os.makedirs(os.path.dirname(MODELS_FILE), exist_ok=True)
        with open(MODELS_FILE, 'w') as f:
            json.dump({"models": []}, f, indent=2)


def load_models_data():
    """Load all models data from the consolidated file."""
    setup_environment()
    with open(MODELS_FILE, 'r') as f:
        return json.load(f)


def save_models_data(models_data):
    """Save all models data to the consolidated file."""
    with open(MODELS_FILE, 'w') as f:
        json.dump(models_data, f, indent=2)


def scan_models_directory():
    """Scan the weights directory for timestamped metadata files and consolidate them."""
    setup_environment()

    # Get all metadata files matching the pattern metadata_*.json in models directories
    # Look in weights/{time_window}/{flare_class}/metadata_*.json
    metadata_files = []
    for time_window in ["24", "48", "72"]:
        for flare_class in ["C", "M", "M5"]:
            # Path to search:
            # weights/{time_window}/{flare_class}/metadata_*.json
            search_path = os.path.join(
                WEIGHTS_BASE_DIR,
                time_window,
                flare_class,
                "metadata_*.json")
            found_files = glob.glob(search_path)
            # Don't include latest metadata
            found_files = [
                f for f in found_files if "metadata_latest.json" not in f]
            metadata_files.extend(found_files)

    # Load existing models data
    models_data = load_models_data()
    models_data["models"] = []

    print(f"Found {len(metadata_files)} model metadata files")

    # Process each metadata file
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Extract key information
            timestamp = metadata.get("timestamp", "unknown")
            flare_class = metadata.get("flare_class", "unknown")
            time_window = metadata.get("time_window", "unknown")
            description = metadata.get("description", "No description")

            # Extract folder structure info if not in metadata
            if time_window == "unknown" or flare_class == "unknown":
                # Extract from path:
                # weights/{time_window}/{flare_class}/metadata_*.json
                parts = metadata_file.split(os.sep)
                if len(parts) >= 3:
                    if time_window == "unknown":
                        # Extract time window from path
                        time_window = parts[-3]
                    if flare_class == "unknown":
                        # Extract flare class from path
                        flare_class = parts[-2]

            # Create model entry
            model_entry = {
                "name": f"{flare_class}_{time_window}_{timestamp}",
                "timestamp": timestamp,
                "description": description,
                "flare_class": flare_class,
                "time_window": time_window,
                "metadata": metadata,
                "file_path": metadata_file
            }

            # Add to models list
            models_data["models"].append(model_entry)

        except Exception as e:
            print(f"Error processing {metadata_file}: {str(e)}")

    # Save consolidated data
    save_models_data(models_data)
    print(
        f"Consolidated {len(models_data['models'])} models into {MODELS_FILE}")

    return models_data


def list_models():
    """List all tracked models with their descriptions."""
    models_data = load_models_data()

    if not models_data["models"]:
        print("No models have been saved yet. Run 'scan' command first.")
        return

    print("\n=== TRACKED MODELS ===")
    print(f"{'FLARE CLASS':<12} {'TIME WINDOW':<12} {'TIMESTAMP':<17} {'DESCRIPTION':<40}")
    print("=" * 80)

    for model in sorted(
        models_data["models"], key=lambda x: (
            x.get(
            "time_window", ""), x.get(
                "flare_class", ""), x.get(
                    "timestamp", ""))):
        print(
            f"{model.get('flare_class', 'unknown'):<12} {model.get('time_window', 'unknown'):<12} {model.get('timestamp', 'unknown'):<17} {model.get('description', '')[:40]:<40}")


def show_model_results(model_name):
    """Show results for a specific model."""
    models_data = load_models_data()

    # Find the model
    model_entry = None
    for model in models_data["models"]:
        if model["name"] == model_name:
            model_entry = model
            break

    if not model_entry:
        # Try searching by timestamp
        for model in models_data["models"]:
            if model["timestamp"] == model_name:
                model_entry = model
                break

    if not model_entry:
        print(f"Model '{model_name}' not found.")
        return

    # Display the model information
    print(f"\n=== MODEL: {model_entry['name']} ===")
    print(f"Flare Class: {model_entry.get('flare_class', 'unknown')}")
    print(f"Time Window: {model_entry.get('time_window', 'unknown')} hours")
    print(f"Timestamp: {model_entry.get('timestamp', 'unknown')}")
    print(f"Description: {model_entry.get('description', 'No description')}")
    print(f"Metadata File: {model_entry.get('file_path', 'unknown')}")
    print("=" * 80)

    # Display architecture details if available
    if "metadata" in model_entry and "model_architecture" in model_entry["metadata"]:
        arch = model_entry["metadata"]["model_architecture"]
        print("\nMODEL ARCHITECTURE:")
        print(
            f"Transformer Blocks: {arch.get('num_transformer_blocks', 'unknown')}")
        print(f"Embedding Dim: {arch.get('embed_dim', 'unknown')}")
        print(f"Attention Heads: {arch.get('num_heads', 'unknown')}")
        print(f"FF Dim: {arch.get('ff_dim', 'unknown')}")
        print(f"Dropout Rate: {arch.get('dropout_rate', 'unknown')}")
        print(f"Total Parameters: {arch.get('total_params', 'unknown')}")

    # Display training info if available
    if "metadata" in model_entry and "training" in model_entry["metadata"]:
        training = model_entry["metadata"]["training"]
        print("\nTRAINING INFORMATION:")
        print(f"Optimizer: {training.get('optimizer', 'unknown')}")
        print(f"Learning Rate: {training.get('learning_rate', 'unknown')}")
        print(f"Batch Size: {training.get('batch_size', 'unknown')}")
        print(
            f"Epochs: {training.get('epochs', 'unknown')} (completed {training.get('actual_epochs', 'unknown')})")
        print(
            f"Training Samples: {training.get('training_samples', 'unknown')}")
        print(f"Final Loss: {training.get('final_loss', 'unknown')}")
        print(f"Final Accuracy: {training.get('final_accuracy', 'unknown')}")
        print(
            f"Training Time: {training.get('training_time_seconds', 'unknown')} seconds")

    # Display results if available
    if "metadata" in model_entry and "results" in model_entry["metadata"]:
        results = model_entry["metadata"]["results"]
        print("\nTEST RESULTS:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")


def generate_comparison(metric="TSS", flare_class=None, horizon=None):
    """Generate comparison of all models for a specific metric, optionally filtered by flare class and horizon."""
    models_data = load_models_data()

    if not models_data["models"]:
        print("No models have been saved yet. Run 'scan' command first.")
        return

    print(f"\n=== MODEL COMPARISON ===")
    print(f"Metric: {metric}")
    if flare_class:
        print(f"Filtered by Flare Class: {flare_class}")
    if horizon:
        print(f"Filtered by Time Horizon: {horizon} hours")
    print("=" * 80)

    comparison_data = []

    for model in models_data["models"]:
        # Apply filters if specified
        if flare_class and model.get("flare_class") != flare_class:
            continue

        if horizon and model.get("time_window") != horizon:
            continue

        # Get results from metadata
        if "metadata" in model and "results" in model["metadata"]:
            results = model["metadata"]["results"]

            # Extract the specific metric
            if metric in results:
                value = results[metric]
                comparison_data.append({
                    "model": model["name"],
                    "timestamp": model.get("timestamp", "unknown"),
                    "flare_class": model.get("flare_class", "unknown"),
                    "time_window": model.get("time_window", "unknown"),
                    "value": value
                })
            else:
                print(
                    f"Warning: Metric {metric} not found for model '{model['name']}'")

    if not comparison_data:
        print("No models found with the specified criteria.")
        return

    # Sort by value
    comparison_data.sort(key=lambda x: x["value"], reverse=True)

    # Print comparison
    print(f"{'FLARE':<6} {'WINDOW':<8} {'TIMESTAMP':<17} {metric:<10}")
    print("-" * 45)

    for item in comparison_data:
        print(
            f"{item['flare_class']:<6} {item['time_window']:<8} {item['timestamp']:<17} {item['value']}")

    # Calculate statistics
    if comparison_data:
        values = [item["value"] for item in comparison_data]
        print("\nSummary Statistics:")
        print(f"  Best: {max(values):.4f} ({comparison_data[0]['model']})")
        print(f"  Worst: {min(values):.4f}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Median: {np.median(values):.4f}")
        print(f"  Std Dev: {np.std(values):.4f}")


def export_to_csv(output_file="model_comparison.csv"):
    """Export all model results to a CSV file for further analysis."""
    models_data = load_models_data()

    if not models_data["models"]:
        print("No models have been saved yet. Run 'scan' command first.")
        return

    # Create rows for the CSV
    rows = []

    for model in models_data["models"]:
        # Get base information
        row = {
            "name": model["name"],
            "timestamp": model.get("timestamp", ""),
            "flare_class": model.get("flare_class", ""),
            "time_window": model.get("time_window", ""),
            "description": model.get("description", "")
        }

        # Add architecture info if available
        if "metadata" in model and "model_architecture" in model["metadata"]:
            arch = model["metadata"]["model_architecture"]
            for key, value in arch.items():
                row[f"arch_{key}"] = value

        # Add training info if available
        if "metadata" in model and "training" in model["metadata"]:
            training = model["metadata"]["training"]
            for key, value in training.items():
                row[f"training_{key}"] = value

        # Add results if available
        if "metadata" in model and "results" in model["metadata"]:
            results = model["metadata"]["results"]
            for key, value in results.items():
                row[f"result_{key}"] = value

        rows.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Track and compare solar flare prediction model results")
    subparsers = parser.add_subparsers(
        dest="command", help="Command to execute")

    # Scan command
    subparsers.add_parser("scan", help="Scan models directory for metadata")

    # List command
    subparsers.add_parser("list", help="List all tracked models")

    # Show command
    show_parser = subparsers.add_parser(
        "show", help="Show results for a specific model")
    show_parser.add_argument("model_name",
                             help="Name or timestamp of the model to show")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare models on a specific metric")
    compare_parser.add_argument("-m", "--metric", default="TSS",
                                help="Metric to compare (default: TSS)")
    compare_parser.add_argument("-c", "--class", dest="flare_class",
                                help="Filter by flare class (e.g., C, M, M5)")
    compare_parser.add_argument(
        "-t",
        "--horizon",
        help="Filter by time horizon in hours (e.g., 24, 48, 72)")

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export all model results to CSV")
    export_parser.add_argument(
        "-o",
        "--output",
        default="model_comparison.csv",
        help="Output CSV file (default: model_comparison.csv)")

    args = parser.parse_args()

    if args.command == "scan":
        scan_models_directory()
    elif args.command == "list":
        list_models()
    elif args.command == "show":
        show_model_results(args.model_name)
    elif args.command == "compare":
        generate_comparison(args.metric, args.flare_class, args.horizon)
    elif args.command == "export":
        export_to_csv(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
