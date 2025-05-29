#!/usr/bin/env python3
"""
Cluster wrapper for EVEREST ablation studies using the current trainer system.
"""

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study variant")
    parser.add_argument("--variant", required=True, help="Ablation variant to train")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--sequence", help="Optional sequence length variant")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--memory-efficient", action="store_true", help="Enable memory efficient training")
    
    args = parser.parse_args()
    
    # Import and run the current trainer
    from trainer import train_ablation_variant
    
    print(f"🚀 Starting ablation experiment: {args.variant}, seed {args.seed}")
    
    results = train_ablation_variant(
        args.variant, 
        args.seed, 
        args.sequence,
        batch_size_override=args.batch_size,
        memory_efficient=args.memory_efficient
    )
    
    if results:
        print(f"✅ Experiment completed successfully!")
        print(f"   Final TSS: {results['final_metrics']['tss']:.4f}")
        print(f"   Final F1: {results['final_metrics']['f1']:.4f}")
    else:
        print(f"❌ Experiment failed!")
        sys.exit(1) 