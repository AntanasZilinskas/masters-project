#!/usr/bin/env python
"""
Full evaluation script for nightly runs.
Evaluates the model on a specified date range of data.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run full evaluation on date range")
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights file"
    )
    parser.add_argument(
        "--split", type=str, required=True,
        help="Date range in format YYYY-MM:YYYY-MM"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--data-s3-url", type=str,
        help="S3 URL for downloading full dataset if needed"
    )
    return parser.parse_args()


def parse_date_range(date_range):
    """Parse date range string into start and end datetime objects."""
    try:
        start_str, end_str = date_range.split(":")
        start_date = datetime.strptime(start_str, "%Y-%m")
        end_date = datetime.strptime(end_str, "%Y-%m")
        return start_date, end_date
    except ValueError:
        print(f"Error: Invalid date range format: {date_range}")
        print("Expected format: YYYY-MM:YYYY-MM")
        sys.exit(1)


def main():
    args = parse_args()
    print(f"Starting full evaluation with weights from {args.weights}")
    
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file {args.weights} does not exist")
        sys.exit(1)
    
    # Parse date range
    start_date, end_date = parse_date_range(args.split)
    print(f"Evaluating on data from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    try:
        model = tf.keras.models.load_model(args.weights)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # In a real implementation, this would load and process actual data
    # For demonstration, we'll use dummy data
    test_samples = 100
    x_test = tf.random.normal((test_samples, 100, 14))
    y_test = tf.random.uniform((test_samples,), maxval=2, dtype=tf.int32)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = model.evaluate(x_test, y_test, verbose=1)
    
    # Get metric names
    metric_names = model.metrics_names
    
    # Create results dictionary
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_weights": args.weights,
        "data_range": args.split,
        "metrics": {name: float(value) for name, value in zip(metric_names, metrics)},
        "samples_evaluated": test_samples
    }
    
    # Save results to file
    output_file = os.path.join(
        args.output_dir,
        f"nightly_{datetime.now().strftime('%Y%m%d')}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 