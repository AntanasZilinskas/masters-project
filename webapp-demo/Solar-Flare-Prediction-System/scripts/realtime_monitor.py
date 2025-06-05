#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_monitor.py
Continuous monitoring script for real-time solar flare prediction data collection.

This script runs continuously and:
1. Collects the latest SHARP data every few hours
2. Backfills labels for older data once flare outcomes are known
3. Maintains a rolling dataset for operational flare prediction

Usage:
    python realtime_monitor.py --interval 12 --output-dir realtime_data
"""

import time
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("realtime_monitor.log")
    ]
)
log = logging.getLogger(__name__)

def run_data_collection(output_dir: str, lookback_hours: int, label_delay_hours: int) -> bool:
    """
    Run a single data collection cycle.
    
    Args:
        output_dir: Directory to store data
        lookback_hours: Hours of SHARP data to collect
        label_delay_hours: Hours to wait before labeling
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            sys.executable, "download_new.py",
            "--realtime",
            "--output-dir", output_dir,
            "--lookback-hours", str(lookback_hours),
            "--label-delay-hours", str(label_delay_hours)
        ]
        
        log.info(f"Running data collection: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            log.info("Data collection completed successfully")
            if result.stdout:
                log.info(f"Output: {result.stdout}")
            return True
        else:
            log.error(f"Data collection failed with return code {result.returncode}")
            if result.stderr:
                log.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        log.error("Data collection timed out after 1 hour")
        return False
    except Exception as e:
        log.error(f"Error running data collection: {e}")
        return False

def cleanup_old_files(output_dir: str, max_age_days: int = 30) -> None:
    """
    Clean up old unlabeled files to prevent disk space issues.
    
    Args:
        output_dir: Directory containing data files
        max_age_days: Maximum age of files to keep
    """
    try:
        output_path = Path(output_dir)
        if not output_path.exists():
            return
            
        cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        # Clean up old unlabeled files (keep labeled files longer)
        unlabeled_files = list(output_path.glob("unlabeled_data_*.csv"))
        cleaned_count = 0
        
        for file_path in unlabeled_files:
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                cleaned_count += 1
                
        if cleaned_count > 0:
            log.info(f"Cleaned up {cleaned_count} old unlabeled files")
            
    except Exception as e:
        log.warning(f"Error during cleanup: {e}")

def monitor_disk_space(output_dir: str, min_free_gb: float = 5.0) -> bool:
    """
    Check if there's enough disk space available.
    
    Args:
        output_dir: Directory to check
        min_free_gb: Minimum free space required in GB
        
    Returns:
        True if enough space, False otherwise
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free / (1024**3)
        
        if free_gb < min_free_gb:
            log.warning(f"Low disk space: {free_gb:.1f} GB free (minimum: {min_free_gb} GB)")
            return False
        else:
            log.debug(f"Disk space OK: {free_gb:.1f} GB free")
            return True
            
    except Exception as e:
        log.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check

def main():
    parser = argparse.ArgumentParser(description="Continuous real-time solar flare data monitoring")
    parser.add_argument("--interval", type=int, default=12, help="Collection interval in hours (default: 12)")
    parser.add_argument("--output-dir", type=str, default="realtime_data", help="Output directory")
    parser.add_argument("--lookback-hours", type=int, default=72, help="Hours of SHARP data to collect")
    parser.add_argument("--label-delay-hours", type=int, default=48, help="Hours to wait before labeling")
    parser.add_argument("--cleanup-days", type=int, default=30, help="Days to keep old files")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="Minimum free disk space in GB")
    parser.add_argument("--max-failures", type=int, default=5, help="Max consecutive failures before stopping")
    args = parser.parse_args()
    
    log.info("Starting real-time solar flare data monitoring")
    log.info(f"Collection interval: {args.interval} hours")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Lookback period: {args.lookback_hours} hours")
    log.info(f"Label delay: {args.label_delay_hours} hours")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    consecutive_failures = 0
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            log.info(f"Starting collection cycle #{cycle_count}")
            
            # Check disk space
            if not monitor_disk_space(args.output_dir, args.min_free_gb):
                log.warning("Insufficient disk space - running cleanup")
                cleanup_old_files(args.output_dir, args.cleanup_days // 2)  # More aggressive cleanup
            
            # Run data collection
            success = run_data_collection(
                args.output_dir,
                args.lookback_hours,
                args.label_delay_hours
            )
            
            if success:
                consecutive_failures = 0
                log.info(f"Cycle #{cycle_count} completed successfully")
                
                # Periodic cleanup
                if cycle_count % 24 == 0:  # Daily cleanup
                    cleanup_old_files(args.output_dir, args.cleanup_days)
                    
            else:
                consecutive_failures += 1
                log.error(f"Cycle #{cycle_count} failed (consecutive failures: {consecutive_failures})")
                
                if consecutive_failures >= args.max_failures:
                    log.error(f"Too many consecutive failures ({consecutive_failures}). Stopping monitor.")
                    break
            
            # Wait for next cycle
            sleep_seconds = args.interval * 3600
            log.info(f"Waiting {args.interval} hours until next collection cycle...")
            time.sleep(sleep_seconds)
            
    except KeyboardInterrupt:
        log.info("Monitoring stopped by user")
    except Exception as e:
        log.error(f"Unexpected error in monitoring loop: {e}")
        raise
    
    log.info("Real-time monitoring stopped")

if __name__ == "__main__":
    main() 