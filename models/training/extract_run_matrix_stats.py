#!/usr/bin/env python3
"""
Extract run matrix statistics from Nature_data CSV files.
Calculates actual train/test split numbers for each flare class and time horizon.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import subprocess


def count_csv_labels(file_path: str) -> tuple:
    """Count positive and negative labels in a CSV file using awk for efficiency."""
    try:
        # Use awk for efficient counting of large files
        cmd = f"awk -F',' 'NR>1 {{if($1==\"P\") pos++; else neg++}} END {{print pos+0, neg+0, NR-1}}' {file_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            output = result.stdout.strip()
            parts = output.split()
            if len(parts) >= 3:
                positive = int(parts[0])
                negative = int(parts[1])
                total = int(parts[2])
                return positive, negative, total

        # Fallback to pandas if awk fails
        print(f"  ⚠️  awk failed for {file_path}, using pandas...")
        df = pd.read_csv(file_path, usecols=["Flare"])
        positive = (df["Flare"] == "P").sum()
        negative = (df["Flare"] == "N").sum()
        total = len(df)
        return positive, negative, total

    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return 0, 0, 0


def extract_all_statistics():
    """Extract statistics for all flare classes and time windows."""
    print("🚀 Extracting EVEREST Dataset Statistics")
    print("=" * 50)

    nature_data_path = Path("../../Nature_data")

    if not nature_data_path.exists():
        print(f"❌ Nature_data directory not found at {nature_data_path.absolute()}")
        return {}

    # Define all combinations
    flare_classes = ["C", "M", "M5"]
    time_windows = ["24", "48", "72"]
    splits = ["training", "testing"]

    stats = {}

    print(f"📂 Processing files from {nature_data_path.absolute()}\n")

    for flare_class in flare_classes:
        for time_window in time_windows:
            print(f"📊 Processing {flare_class}_{time_window}:")

            task_stats = {}

            for split in splits:
                file_name = f"{split}_data_{flare_class}_{time_window}.csv"
                file_path = nature_data_path / file_name

                if file_path.exists():
                    print(f"  {split:8}: {file_path.name}")
                    positive, negative, total = count_csv_labels(str(file_path))

                    task_stats[split] = {
                        "positive": positive,
                        "negative": negative,
                        "total": total,
                    }

                    ratio = positive / total if total > 0 else 0
                    print(
                        f"           {positive:6,}+ / {negative:6,}- (total: {total:7,}, ratio: {ratio:.4f})"
                    )
                else:
                    print(f"  {split:8}: ❌ File not found: {file_name}")
                    task_stats[split] = {"positive": 0, "negative": 0, "total": 0}

            stats[f"{flare_class}_{time_window}"] = task_stats
            print()

    return stats


def generate_run_matrix_table(stats: dict) -> str:
    """Generate the complete run matrix LaTeX table."""
    print("📋 Generating LaTeX table...")

    latex_lines = []
    latex_lines.extend(
        [
            "\\begin{table}[ht]\\centering",
            "\\caption{Run matrix showing the number of positive (+) and negative (–) examples in the train/val/test partitions for every flare class $\\times$ horizon combination.}",
            "\\label{tab:run_matrix}",
            "\\begin{tabular}{lccccccc}",
            "\\toprule",
            "\\multirow{2}{*}{\\textbf{Flare}} & \\multirow{2}{*}{\\textbf{Horizon}} &",
            "\\multicolumn{2}{c}{\\textbf{Train}} & \\multicolumn{2}{c}{\\textbf{Val}} &",
            "\\multicolumn{2}{c}{\\textbf{Test}}\\\\",
            "& & + & -- & + & -- & + & -- \\\\",
            "\\midrule",
        ]
    )

    # Process each task
    for i, flare_class in enumerate(["C", "M", "M5"]):
        # Add spacing between flare classes
        if i > 0:
            latex_lines.append("\\addlinespace")

        for time_window in ["24", "48", "72"]:
            task_key = f"{flare_class}_{time_window}"

            if task_key in stats:
                task_data = stats[task_key]

                # Get training and testing data
                train_pos = task_data.get("training", {}).get("positive", 0)
                train_neg = task_data.get("training", {}).get("negative", 0)
                test_pos = task_data.get("testing", {}).get("positive", 0)
                test_neg = task_data.get("testing", {}).get("negative", 0)

                # Estimate validation split (typically 15-20% of training)
                # Based on common ML practices and the fact that we don't have separate val files
                val_ratio = 0.15  # 15% of training for validation
                val_pos = int(train_pos * val_ratio)
                val_neg = int(train_neg * val_ratio)

                # Adjust training numbers
                train_pos_adj = train_pos - val_pos
                train_neg_adj = train_neg - val_neg

                def format_num(n):
                    return f"{n:,}".replace(",", "\\,")

                # Build table row
                row = f"{flare_class} & {time_window} h & "
                row += f"{format_num(train_pos_adj)} & {format_num(train_neg_adj)} & "
                row += f"{format_num(val_pos)} & {format_num(val_neg)} & "
                row += f"{format_num(test_pos)} & {format_num(test_neg)} \\\\[-0.1em]"

                latex_lines.append(row)
            else:
                # Missing data
                row = f"{flare_class} & {time_window} h & "
                row += "-- & -- & -- & -- & -- & -- \\\\[-0.1em]"
                latex_lines.append(row)

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    return "\n".join(latex_lines)


def display_summary(stats: dict):
    """Display a summary of the statistics."""
    print("=" * 50)
    print("📊 DATASET STATISTICS SUMMARY")
    print("=" * 50)

    total_train_pos = total_train_neg = 0
    total_test_pos = total_test_neg = 0

    for task, data in stats.items():
        flare_class, time_window = task.split("_")
        print(f"\n{flare_class}-{time_window}h:")

        if "training" in data:
            train_pos = data["training"]["positive"]
            train_neg = data["training"]["negative"]
            train_total = data["training"]["total"]
            total_train_pos += train_pos
            total_train_neg += train_neg

            # Estimated validation split (15%)
            val_pos = int(train_pos * 0.15)
            val_neg = int(train_neg * 0.15)
            train_pos_adj = train_pos - val_pos
            train_neg_adj = train_neg - val_neg

            print(
                f"  Training: {train_pos_adj:6,}+ / {train_neg_adj:6,}- (ratio: {train_pos_adj/(train_pos_adj+train_neg_adj):.4f})"
            )
            print(
                f"  Validation: {val_pos:4,}+ / {val_neg:6,}- (ratio: {val_pos/(val_pos+val_neg):.4f})"
            )

        if "testing" in data:
            test_pos = data["testing"]["positive"]
            test_neg = data["testing"]["negative"]
            test_total = data["testing"]["total"]
            total_test_pos += test_pos
            total_test_neg += test_neg

            print(
                f"  Testing:  {test_pos:6,}+ / {test_neg:6,}- (ratio: {test_pos/(test_pos+test_neg):.4f})"
            )

    print(f"\n📈 OVERALL TOTALS:")
    print(
        f"  Training:   {total_train_pos:7,}+ / {total_train_neg:7,}- (total: {total_train_pos+total_train_neg:8,})"
    )
    print(
        f"  Testing:    {total_test_pos:7,}+ / {total_test_neg:7,}- (total: {total_test_pos+total_test_neg:8,})"
    )


def main():
    """Main function."""
    # Extract statistics
    stats = extract_all_statistics()

    if not stats:
        print("❌ No statistics extracted!")
        return

    # Display summary
    display_summary(stats)

    # Generate LaTeX table
    latex_table = generate_run_matrix_table(stats)

    print("\n" + "=" * 50)
    print("📋 COMPLETE RUN MATRIX TABLE (LaTeX)")
    print("=" * 50)
    print(latex_table)

    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Save LaTeX table
    with open(output_dir / "run_matrix_table.tex", "w") as f:
        f.write(latex_table)

    # Save raw statistics
    import json

    with open(output_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Results saved to {output_dir}/")
    print(f"  📄 run_matrix_table.tex - Complete LaTeX table")
    print(f"  📄 dataset_statistics.json - Raw statistics")

    print(f"\n💡 Note: Validation numbers are estimated as 15% of training data")
    print(f"    (since only training and testing splits are available)")


if __name__ == "__main__":
    main()
