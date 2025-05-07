#!/usr/bin/env python
"""
Utility script to migrate existing models to the new models/trained_models directory structure
"""

import os
import sys
import shutil
import json
import numpy as np
from datetime import datetime
import argparse

def migrate_model(source_dir, target_dir='models/trained_models', verbose=True):
    """
    Migrate a model from source directory to target directory with new structure
    
    Args:
        source_dir: Source directory containing model
        target_dir: Target base directory (models/trained_models by default)
        verbose: Whether to print progress
    
    Returns:
        Path to the new model directory
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return None
        
    if not os.path.exists(os.path.join(source_dir, 'metadata.json')):
        print(f"Source directory {source_dir} does not contain metadata.json.")
        return None
        
    # Load metadata
    with open(os.path.join(source_dir, 'metadata.json'), 'r') as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding metadata.json in {source_dir}")
            return None
    
    # Extract model information
    model_name = metadata.get('model_name', 'EVEREST')
    flare_class = metadata.get('flare_class', '')
    time_window = metadata.get('time_window', '')
    version = metadata.get('version', '1')
    
    # If time_window is missing, try to extract from directory name
    if not time_window and '-' in os.path.basename(source_dir):
        parts = os.path.basename(source_dir).split('-')
        if len(parts) > 1 and parts[-1].endswith('h'):
            time_window = parts[-1].rstrip('h')
    
    # Create target directory path
    target_subdir = f"{model_name}-v{version}-{flare_class}-{time_window}h"
    target_model_dir = os.path.join(target_dir, target_subdir)
    
    # Create target directory
    os.makedirs(target_model_dir, exist_ok=True)
    if verbose:
        print(f"Created target directory: {target_model_dir}")
    
    # Copy files
    files_to_copy = ['model_weights.weights.h5', 'metadata.json']
    for file in files_to_copy:
        source_file = os.path.join(source_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(target_model_dir, file))
            if verbose:
                print(f"Copied {file}")
        else:
            print(f"Warning: {file} not found in {source_dir}")
    
    # Convert history to numpy if it's in the metadata
    if 'history' in metadata:
        history_dict = metadata.pop('history')
        np.save(os.path.join(target_model_dir, 'history.npy'), history_dict)
        # Update metadata to reference history keys
        metadata['history_keys'] = list(history_dict.keys())
        
        # Write updated metadata
        with open(os.path.join(target_model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if verbose:
            print(f"Extracted history to numpy file")
    
    # Copy any validation data if it exists
    validation_files = ['val_logits.npy', 'val_labels.npy', 'val_evidential.npy', 'val_evt.npy']
    for file in validation_files:
        source_file = os.path.join(source_dir, file)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(target_model_dir, file))
            if verbose:
                print(f"Copied validation data: {file}")
    
    if verbose:
        print(f"Successfully migrated model to {target_model_dir}")
    
    return target_model_dir

def scan_and_migrate_models(verbose=True):
    """
    Scan for existing models and migrate them to the new directory structure
    
    Args:
        verbose: Whether to print progress
    
    Returns:
        List of migrated model directories
    """
    # Ensure target directory exists
    target_dir = 'models/trained_models'
    os.makedirs(target_dir, exist_ok=True)
    
    migrated_models = []
    
    # Common model directories
    model_dirs = [
        os.path.join('models', 'EVEREST'),
        os.path.join('models', 'SolarKnowledge')
    ]
    
    # Scan for flare class subdirectories in model directories
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            continue
        
        for flare_class in os.listdir(model_dir):
            flare_dir = os.path.join(model_dir, flare_class)
            if os.path.isdir(flare_dir):
                if verbose:
                    print(f"Migrating {flare_dir}...")
                migrated = migrate_model(flare_dir, target_dir, verbose)
                if migrated:
                    migrated_models.append(migrated)
    
    # Scan for SolarKnowledge versioned directories
    for item in os.listdir('models'):
        if item.startswith('SolarKnowledge-v') and os.path.isdir(os.path.join('models', item)):
            source_dir = os.path.join('models', item)
            if verbose:
                print(f"Migrating {source_dir}...")
            migrated = migrate_model(source_dir, target_dir, verbose)
            if migrated:
                migrated_models.append(migrated)
    
    if verbose:
        print(f"Migrated {len(migrated_models)} models to {target_dir}")
    
    return migrated_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate models to the new directory structure')
    parser.add_argument('--source', type=str, help='Specific source directory to migrate')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()
    
    if args.source:
        # Migrate specific model
        migrate_model(args.source, verbose=not args.quiet)
    else:
        # Scan and migrate all models
        scan_and_migrate_models(verbose=not args.quiet) 