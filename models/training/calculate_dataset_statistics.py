#!/usr/bin/env python3
"""
Calculate dataset statistics for the run matrix table.
Extracts actual train/validation/test split numbers for each flare class and time horizon.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

def find_data_files() -> Dict[str, str]:
    """Find dataset files in common locations."""
    print("🔍 Searching for dataset files...")
    
    # Common data locations to check
    possible_data_dirs = [
        "../../data",
        "../../datasets", 
        "../data",
        "../datasets",
        "../../data_pipeline",
        "../../Nature_data",
        "../Nature_data",
        "data",
        "datasets"
    ]
    
    found_files = {}
    
    for data_dir in possible_data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            continue
            
        print(f"  Checking: {data_path.absolute()}")
        
        # Look for split files
        for split in ['train', 'val', 'test']:
            split_file = data_path / f"{split}.csv"
            if split_file.exists():
                found_files[split] = str(split_file)
                print(f"    ✅ Found {split}.csv")
        
        # Look for metadata files
        meta_file = data_path / "meta.json"
        if meta_file.exists():
            found_files['meta'] = str(meta_file)
            print(f"    ✅ Found meta.json")
        
        # Look for specific test data files
        for pattern in ["*test*.csv", "*M5*.csv", "*C*.csv", "*M*.csv"]:
            matching_files = list(data_path.glob(pattern))
            for file in matching_files:
                if file.stem not in found_files:
                    found_files[file.stem] = str(file)
                    print(f"    ✅ Found {file.name}")
    
    return found_files

def load_metadata_if_exists() -> Optional[Dict]:
    """Load metadata from data pipeline if it exists."""
    data_files = find_data_files()
    
    if 'meta' in data_files:
        print(f"📋 Loading metadata from {data_files['meta']}")
        try:
            with open(data_files['meta'], 'r') as f:
                meta = json.load(f)
            return meta
        except Exception as e:
            print(f"  ⚠️  Error loading metadata: {e}")
    
    return None

def analyze_csv_file(file_path: str) -> Dict[str, int]:
    """Analyze a CSV file to extract flare statistics."""
    try:
        df = pd.read_csv(file_path)
        print(f"  📊 Analyzing {Path(file_path).name}: {len(df)} rows")
        
        stats = {}
        
        # Look for flare columns in different naming conventions
        flare_columns = [col for col in df.columns if 'flare' in col.lower()]
        
        if not flare_columns:
            # Look for alternative patterns
            flare_columns = [col for col in df.columns if any(x in col.upper() for x in ['C_24', 'M_24', 'M5_24', 'C_48', 'M_48', 'M5_48', 'C_72', 'M_72', 'M5_72'])]
        
        print(f"    🎯 Found flare columns: {flare_columns}")
        
        for col in flare_columns:
            if col in df.columns:
                positive = int(df[col].sum())
                negative = len(df) - positive
                stats[col] = {
                    'positive': positive,
                    'negative': negative,
                    'total': len(df)
                }
                print(f"      {col}: {positive}+ / {negative}- (total: {len(df)})")
        
        return stats
        
    except Exception as e:
        print(f"  ❌ Error analyzing {file_path}: {e}")
        return {}

def extract_from_metadata(meta: Dict) -> Dict[str, Dict[str, int]]:
    """Extract statistics from metadata if available."""
    print("📋 Extracting from metadata...")
    
    stats = {}
    
    if 'class_distribution' in meta:
        for split in ['train', 'validation', 'test']:
            if split in meta['class_distribution']:
                split_data = meta['class_distribution'][split]
                
                for flare_col, positive_count in split_data.items():
                    # Convert from metadata naming to our format
                    flare_name = flare_col.replace('flare_', '').replace('_24h', '_24').replace('_48h', '_48').replace('_72h', '_72')
                    
                    if split == 'validation':
                        split_key = 'val'
                    else:
                        split_key = split
                    
                    key = f"{flare_name}_{split_key}"
                    
                    # Get total count for negative calculation
                    total_count = meta.get('data_counts', {}).get(split, 0)
                    negative_count = total_count - positive_count
                    
                    stats[key] = {
                        'positive': positive_count,
                        'negative': negative_count,
                        'total': total_count
                    }
                    
                    print(f"  {key}: {positive_count}+ / {negative_count}- (total: {total_count})")
    
    return stats

def calculate_statistics_from_files() -> Dict[str, Dict[str, int]]:
    """Calculate statistics directly from data files."""
    print("📊 Calculating statistics from data files...")
    
    data_files = find_data_files()
    all_stats = {}
    
    # Analyze each split file
    for split in ['train', 'val', 'test']:
        if split in data_files:
            file_stats = analyze_csv_file(data_files[split])
            
            # Reorganize by flare class and time window
            for flare_col, stats in file_stats.items():
                # Parse flare column name to extract class and time window
                if 'flare' in flare_col.lower():
                    # Remove 'flare_' prefix and parse
                    clean_name = flare_col.lower().replace('flare_', '')
                    
                    # Extract flare class and time window
                    if 'm5' in clean_name:
                        flare_class = 'M5'
                        time_window = clean_name.replace('m5_', '').replace('h', '')
                    elif 'm_' in clean_name:
                        flare_class = 'M'
                        time_window = clean_name.replace('m_', '').replace('h', '')
                    elif 'c_' in clean_name:
                        flare_class = 'C'
                        time_window = clean_name.replace('c_', '').replace('h', '')
                    else:
                        continue
                    
                    key = f"{flare_class}_{time_window}_{split}"
                    all_stats[key] = stats
    
    return all_stats

def generate_latex_table(stats: Dict[str, Dict[str, int]]) -> str:
    """Generate LaTeX table for the run matrix."""
    print("📋 Generating LaTeX table...")
    
    # Organize data by flare class and time window
    table_data = {}
    
    for key, stat in stats.items():
        parts = key.split('_')
        if len(parts) >= 3:
            flare_class = parts[0]
            time_window = parts[1]
            split = parts[2]
            
            task_key = f"{flare_class}_{time_window}"
            if task_key not in table_data:
                table_data[task_key] = {}
            
            table_data[task_key][split] = stat
    
    # Generate LaTeX
    latex_lines = []
    latex_lines.extend([
        "\\begin{table}[ht]\\centering",
        "\\caption{Run matrix showing the number of positive (+) and negative (–) examples in the train/val/test partitions for every flare class $\\times$ horizon combination.}",
        "\\label{tab:run_matrix}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Flare}} & \\multirow{2}{*}{\\textbf{Horizon}} &",
        "\\multicolumn{2}{c}{\\textbf{Train}} & \\multicolumn{2}{c}{\\textbf{Val}} &",
        "\\multicolumn{2}{c}{\\textbf{Test}}\\\\",
        "& & + & -- & + & -- & + & -- \\\\",
        "\\midrule"
    ])
    
    # Sort tasks
    ordered_tasks = []
    for flare in ['C', 'M', 'M5']:
        for time in ['24', '48', '72']:
            task_key = f"{flare}_{time}"
            if task_key in table_data:
                ordered_tasks.append(task_key)
    
    # Add table rows
    for i, task_key in enumerate(ordered_tasks):
        flare_class, time_window = task_key.split('_')
        data = table_data[task_key]
        
        # Add spacing between flare classes
        if i > 0 and task_key.split('_')[0] != ordered_tasks[i-1].split('_')[0]:
            latex_lines.append("\\addlinespace")
        
        # Get statistics for each split
        train_pos = data.get('train', {}).get('positive', 0)
        train_neg = data.get('train', {}).get('negative', 0)
        val_pos = data.get('val', {}).get('positive', 0)
        val_neg = data.get('val', {}).get('negative', 0)
        test_pos = data.get('test', {}).get('positive', 0)
        test_neg = data.get('test', {}).get('negative', 0)
        
        # Format numbers with thousands separators
        def format_num(n):
            return f"{n:,}".replace(",", "\\,")
        
        # Build row
        row = f"{flare_class} & {time_window} h & "
        row += f"{format_num(train_pos)} & {format_num(train_neg)} & "
        row += f"{format_num(val_pos)} & {format_num(val_neg)} & "
        row += f"{format_num(test_pos)} & {format_num(test_neg)} \\\\[-0.1em]"
        
        latex_lines.append(row)
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_lines)

def main():
    """Main function to calculate and display dataset statistics."""
    print("🚀 EVEREST Dataset Statistics Calculator")
    print("=" * 50)
    
    # Try to load from metadata first
    meta = load_metadata_if_exists()
    
    if meta and 'class_distribution' in meta:
        print("✅ Using metadata from data pipeline")
        stats = extract_from_metadata(meta)
    else:
        print("📊 Analyzing data files directly")
        stats = calculate_statistics_from_files()
    
    if not stats:
        print("❌ No dataset statistics found!")
        print("\n💡 Suggestions:")
        print("  1. Check if data files exist in expected locations")
        print("  2. Run the data pipeline to generate splits")
        print("  3. Verify file formats and column names")
        return
    
    # Display statistics
    print("\n" + "=" * 50)
    print("📊 DATASET STATISTICS")
    print("=" * 50)
    
    # Group by task for display
    tasks = {}
    for key, stat in stats.items():
        parts = key.split('_')
        if len(parts) >= 3:
            task = f"{parts[0]}_{parts[1]}"
            split = parts[2]
            if task not in tasks:
                tasks[task] = {}
            tasks[task][split] = stat
    
    for task, splits in sorted(tasks.items()):
        print(f"\n{task.replace('_', '-')}:")
        for split in ['train', 'val', 'test']:
            if split in splits:
                pos = splits[split]['positive']
                neg = splits[split]['negative']
                total = splits[split]['total']
                ratio = pos / total if total > 0 else 0
                print(f"  {split:5}: {pos:6,}+ / {neg:6,}- (total: {total:7,}, ratio: {ratio:.4f})")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(stats)
    
    print("\n" + "=" * 50)
    print("📋 LATEX TABLE FOR PAPER")
    print("=" * 50)
    print(latex_table)
    
    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save statistics as JSON
    with open(output_dir / "dataset_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save LaTeX table
    with open(output_dir / "run_matrix_table.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"\n✅ Results saved to {output_dir}/")
    print(f"  📄 dataset_statistics.json")
    print(f"  📄 run_matrix_table.tex")

if __name__ == "__main__":
    main() 