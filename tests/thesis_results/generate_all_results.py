#!/usr/bin/env python3
"""
EVEREST Thesis Results Generator - Main Script

This script coordinates the complete generation of all results needed for the thesis.
Run this from the thesis_generation folder to generate all tables, figures, and analysis.

Usage:
    python generate_all_results.py [--mode MODE]

Modes:
    - simple: Use simple generator (no torch dependencies) - DEFAULT
    - full: Use full generator (requires torch and actual data)
    - extract: Only extract actual results
    - validate: Only validate completeness
"""

import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from simple_publication_generator import SimplePublicationGenerator


def main():
    """Main function to generate all thesis results."""
    parser = argparse.ArgumentParser(description='Generate EVEREST thesis results')
    parser.add_argument('--mode', choices=['simple', 'full', 'extract', 'validate'], 
                       default='simple', help='Generation mode')
    parser.add_argument('--output-dir', default='output', 
                       help='Output directory name')
    
    args = parser.parse_args()
    
    print("🎓 EVEREST Thesis Results Generator")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Working directory: {os.getcwd()}")
    
    if args.mode == 'simple':
        print("\n🚀 Running simple publication generator...")
        print("   (No torch dependencies required)")
        
        generator = SimplePublicationGenerator()
        generator.output_dir = Path(args.output_dir)
        generator.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (generator.output_dir / "tables").mkdir(exist_ok=True)
        (generator.output_dir / "figures").mkdir(exist_ok=True)
        (generator.output_dir / "data").mkdir(exist_ok=True)
        
        generator.generate_all_results()
        
        print(f"\n✅ Simple generation complete!")
        print(f"📁 Results saved to: {generator.output_dir}")
        
    elif args.mode == 'full':
        print("\n🚀 Running full thesis generator...")
        print("   (Requires torch and actual experimental data)")
        
        try:
            # Import the full generator
            sys.path.append('.')
            from generate_thesis_results import ThesisResultsOrchestrator
            
            orchestrator = ThesisResultsOrchestrator(output_dir=args.output_dir)
            orchestrator.generate_all_results()
            
        except ImportError as e:
            print(f"❌ Error: Cannot import full generator: {e}")
            print("💡 Try running with --mode simple instead")
            sys.exit(1)
            
    elif args.mode == 'extract':
        print("\n🔍 Extracting actual experimental results...")
        
        try:
            from extract_actual_results import ActualResultsExtractor
            
            extractor = ActualResultsExtractor()
            extractor.output_dir = Path(args.output_dir) / "actual_results"
            extractor.extract_all_results()
            
        except ImportError as e:
            print(f"❌ Error: Cannot import extractor: {e}")
            sys.exit(1)
            
    elif args.mode == 'validate':
        print("\n✅ Validating thesis results completeness...")
        
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"❌ Output directory {output_path} does not exist")
            sys.exit(1)
            
        # Check for required files
        required_tables = ['main_performance_table.tex', 'run_matrix_table.tex', 'ablation_table.tex']
        required_figures = ['roc_tss_curves.pdf', 'reliability_diagrams.pdf', 'attention_heatmaps.pdf']
        
        tables_dir = output_path / "tables"
        figures_dir = output_path / "figures"
        
        missing_files = []
        
        for table in required_tables:
            if not (tables_dir / table).exists():
                missing_files.append(f"tables/{table}")
                
        for figure in required_figures:
            if not (figures_dir / figure).exists():
                missing_files.append(f"figures/{figure}")
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            print("💡 Run with --mode simple to generate all files")
        else:
            print("✅ All required files present!")
            print("🎉 Thesis results are complete!")
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("1. Review generated tables and figures")
    print("2. Replace simulated data with actual results (if available)")
    print("3. Integrate LaTeX tables into thesis document")
    print("4. Include PDF figures in thesis")
    print("5. Perform final quality check")


if __name__ == "__main__":
    main() 