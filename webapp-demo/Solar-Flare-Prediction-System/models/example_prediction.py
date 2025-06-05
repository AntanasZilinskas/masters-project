#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_prediction.py
Example script showing how to integrate real-time SHARP data with ML models
for operational solar flare prediction.

This demonstrates:
1. Loading the latest unlabeled SHARP data
2. Applying preprocessing (scaling, padding)
3. Making predictions with a trained model
4. Generating forecast outputs
5. Validating predictions against actual outcomes

Usage:
    python example_prediction.py --data-dir realtime_data --model-path trained_model.pkl
"""

import pandas as pd
import numpy as np
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# SHARP features (same as in download_new.py)
FEATURES = [
    "USFLUX", "TOTUSJH", "TOTUSJZ", "MEANALP", "R_VALUE",
    "TOTPOT", "SAVNCPP", "AREA_ACR", "ABSNJZH",
]

def load_latest_data(data_dir: str) -> pd.DataFrame:
    """
    Load the most recent unlabeled SHARP data.
    
    Args:
        data_dir: Directory containing real-time data files
        
    Returns:
        DataFrame with latest SHARP data
    """
    data_path = Path(data_dir)
    
    # Find the most recent unlabeled file
    unlabeled_files = list(data_path.glob("unlabeled_data_*.csv"))
    if not unlabeled_files:
        raise FileNotFoundError(f"No unlabeled data files found in {data_dir}")
    
    # Sort by timestamp in filename
    latest_file = sorted(unlabeled_files)[-1]
    log.info(f"Loading data from {latest_file}")
    
    df = pd.read_csv(latest_file)
    df['DATE__OBS'] = pd.to_datetime(df['DATE__OBS'])
    
    log.info(f"Loaded {len(df)} samples from {df['NOAA_AR'].nunique()} active regions")
    log.info(f"Data time range: {df['DATE__OBS'].min()} to {df['DATE__OBS'].max()}")
    
    return df

def apply_scaling(df: pd.DataFrame, scales_file: str) -> pd.DataFrame:
    """
    Apply min-max scaling using saved parameters from training.
    
    Args:
        df: DataFrame to scale
        scales_file: JSON file with scaling parameters
        
    Returns:
        Scaled DataFrame
    """
    with open(scales_file, 'r') as f:
        scales = json.load(f)
    
    df_scaled = df.copy()
    
    for feature in FEATURES:
        if feature in scales:
            min_val, max_val = scales[feature]
            span = max_val - min_val if max_val > min_val else 1.0
            df_scaled[feature] = 2 * (df_scaled[feature] - min_val) / span - 1
        else:
            log.warning(f"No scaling parameters found for feature {feature}")
    
    return df_scaled

def create_sequences(df: pd.DataFrame, window_hours: int = 24) -> np.ndarray:
    """
    Create time sequences for each active region.
    
    Args:
        df: Scaled SHARP data
        window_hours: Length of time window in hours
        
    Returns:
        Array of sequences [n_sequences, timesteps, features]
    """
    sequences = []
    
    for noaa_ar in df['NOAA_AR'].unique():
        ar_data = df[df['NOAA_AR'] == noaa_ar].sort_values('DATE__OBS')
        
        if len(ar_data) < 2:  # Need at least 2 timesteps
            continue
            
        # Create sequence for this AR
        feature_data = ar_data[FEATURES].values
        sequences.append(feature_data)
    
    if not sequences:
        return np.array([])
    
    # Pad sequences to same length (use the longest sequence)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros at the beginning
            padding = np.zeros((max_len - len(seq), len(FEATURES)))
            padded_seq = np.vstack([padding, seq])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

def make_predictions(sequences: np.ndarray, model_path: str) -> np.ndarray:
    """
    Make flare predictions using a trained model.
    
    Args:
        sequences: Input sequences [n_sequences, timesteps, features]
        model_path: Path to trained model file
        
    Returns:
        Predictions array [n_sequences, n_classes]
    """
    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    log.info(f"Loaded model from {model_path}")
    log.info(f"Making predictions for {len(sequences)} sequences")
    
    # Make predictions
    predictions = model.predict_proba(sequences.reshape(len(sequences), -1))
    
    return predictions

def generate_forecast(df: pd.DataFrame, predictions: np.ndarray, 
                     forecast_hours: int = 24) -> pd.DataFrame:
    """
    Generate forecast DataFrame with predictions.
    
    Args:
        df: Original SHARP data
        predictions: Model predictions [n_sequences, n_classes]
        forecast_hours: Forecast horizon in hours
        
    Returns:
        DataFrame with forecast information
    """
    # Get the latest timestamp for each AR
    latest_by_ar = df.groupby('NOAA_AR')['DATE__OBS'].max().reset_index()
    
    if len(latest_by_ar) != len(predictions):
        log.warning(f"Mismatch: {len(latest_by_ar)} ARs but {len(predictions)} predictions")
        min_len = min(len(latest_by_ar), len(predictions))
        latest_by_ar = latest_by_ar.iloc[:min_len]
        predictions = predictions[:min_len]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'NOAA_AR': latest_by_ar['NOAA_AR'],
        'Last_Observation': latest_by_ar['DATE__OBS'],
        'Forecast_Start': latest_by_ar['DATE__OBS'],
        'Forecast_End': latest_by_ar['DATE__OBS'] + timedelta(hours=forecast_hours),
        'C_Flare_Probability': predictions[:, 0] if predictions.shape[1] > 0 else 0,
        'M_Flare_Probability': predictions[:, 1] if predictions.shape[1] > 1 else 0,
        'M5_Flare_Probability': predictions[:, 2] if predictions.shape[1] > 2 else 0,
        'Generated_At': datetime.now(timezone.utc)
    })
    
    # Add risk categories
    forecast_df['Risk_Level'] = 'LOW'
    forecast_df.loc[forecast_df['C_Flare_Probability'] > 0.5, 'Risk_Level'] = 'MODERATE'
    forecast_df.loc[forecast_df['M_Flare_Probability'] > 0.3, 'Risk_Level'] = 'HIGH'
    forecast_df.loc[forecast_df['M5_Flare_Probability'] > 0.1, 'Risk_Level'] = 'EXTREME'
    
    return forecast_df

def save_forecast(forecast_df: pd.DataFrame, output_dir: str) -> str:
    """
    Save forecast to file with timestamp.
    
    Args:
        forecast_df: Forecast DataFrame
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"flare_forecast_{timestamp}.csv"
    
    forecast_df.to_csv(filename, index=False)
    log.info(f"Saved forecast to {filename}")
    
    return str(filename)

def validate_predictions(data_dir: str, forecast_file: str, 
                        validation_hours: int = 48) -> pd.DataFrame:
    """
    Validate predictions against actual flare outcomes (if available).
    
    Args:
        data_dir: Directory with labeled data
        forecast_file: Path to forecast file
        validation_hours: Hours to look ahead for validation
        
    Returns:
        DataFrame with validation results
    """
    # Load forecast
    forecast_df = pd.read_csv(forecast_file)
    forecast_df['Forecast_Start'] = pd.to_datetime(forecast_df['Forecast_Start'])
    forecast_df['Forecast_End'] = pd.to_datetime(forecast_df['Forecast_End'])
    
    # Look for labeled data files that might contain validation outcomes
    data_path = Path(data_dir)
    labeled_files = list(data_path.glob("labeled_data_*.csv"))
    
    if not labeled_files:
        log.warning("No labeled data available for validation")
        return pd.DataFrame()
    
    # Load recent labeled data
    validation_results = []
    
    for labeled_file in labeled_files:
        try:
            labeled_df = pd.read_csv(labeled_file)
            labeled_df['DATE__OBS'] = pd.to_datetime(labeled_df['DATE__OBS'])
            
            # Check each forecast
            for _, forecast_row in forecast_df.iterrows():
                ar = forecast_row['NOAA_AR']
                start_time = forecast_row['Forecast_Start']
                end_time = forecast_row['Forecast_End']
                
                # Find actual outcomes for this AR in the forecast window
                ar_data = labeled_df[
                    (labeled_df['NOAA_AR'] == ar) &
                    (labeled_df['DATE__OBS'] >= start_time) &
                    (labeled_df['DATE__OBS'] <= end_time)
                ]
                
                if not ar_data.empty:
                    # Check if any positive flares occurred
                    actual_flare = (ar_data['Flare'] == 'P').any()
                    
                    validation_results.append({
                        'NOAA_AR': ar,
                        'Forecast_Start': start_time,
                        'Predicted_Probability': forecast_row['C_Flare_Probability'],
                        'Actual_Flare': actual_flare,
                        'Validation_File': labeled_file.name
                    })
                    
        except Exception as e:
            log.warning(f"Error processing {labeled_file}: {e}")
            continue
    
    if validation_results:
        validation_df = pd.DataFrame(validation_results)
        log.info(f"Validated {len(validation_df)} predictions")
        return validation_df
    else:
        log.info("No validation data available yet")
        return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser(description="Generate solar flare predictions from real-time data")
    parser.add_argument("--data-dir", type=str, default="realtime_data", 
                       help="Directory containing real-time data")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--scales-file", type=str, default="scales_C_24.json",
                       help="JSON file with scaling parameters")
    parser.add_argument("--output-dir", type=str, default="forecasts",
                       help="Output directory for forecasts")
    parser.add_argument("--window-hours", type=int, default=24,
                       help="Time window for sequences")
    parser.add_argument("--forecast-hours", type=int, default=24,
                       help="Forecast horizon in hours")
    parser.add_argument("--validate", action="store_true",
                       help="Validate previous predictions")
    args = parser.parse_args()
    
    try:
        # Load latest data
        df = load_latest_data(args.data_dir)
        
        if df.empty:
            log.warning("No data available for prediction")
            return
        
        # Apply scaling
        if Path(args.scales_file).exists():
            df_scaled = apply_scaling(df, args.scales_file)
        else:
            log.warning(f"Scaling file {args.scales_file} not found. Using raw data.")
            df_scaled = df
        
        # Create sequences
        sequences = create_sequences(df_scaled, args.window_hours)
        
        if len(sequences) == 0:
            log.warning("No valid sequences created for prediction")
            return
        
        # Make predictions
        predictions = make_predictions(sequences, args.model_path)
        
        # Generate forecast
        forecast_df = generate_forecast(df, predictions, args.forecast_hours)
        
        # Display results
        log.info("\n" + "="*60)
        log.info("SOLAR FLARE FORECAST")
        log.info("="*60)
        
        for _, row in forecast_df.iterrows():
            log.info(f"AR {row['NOAA_AR']:>5}: "
                    f"C={row['C_Flare_Probability']:.3f} "
                    f"M={row['M_Flare_Probability']:.3f} "
                    f"M5={row['M5_Flare_Probability']:.3f} "
                    f"[{row['Risk_Level']}]")
        
        log.info("="*60)
        
        # Save forecast
        forecast_file = save_forecast(forecast_df, args.output_dir)
        
        # Validation (if requested)
        if args.validate:
            validation_df = validate_predictions(args.data_dir, forecast_file)
            if not validation_df.empty:
                accuracy = (validation_df['Predicted_Probability'] > 0.5) == validation_df['Actual_Flare']
                log.info(f"Validation accuracy: {accuracy.mean():.3f}")
        
        log.info("Prediction complete!")
        
    except Exception as e:
        log.error(f"Error in prediction pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 