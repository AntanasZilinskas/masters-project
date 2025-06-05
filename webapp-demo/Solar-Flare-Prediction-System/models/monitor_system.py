#!/usr/bin/env python3
"""
monitor_system.py
System monitoring script for the solar flare prediction deployment.

This script checks:
1. Model availability and versions
2. Data file freshness
3. Prediction output status
4. System health metrics
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

def check_models():
    """Check available models and their versions."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {"status": "error", "message": "Models directory not found"}
    
    models = []
    for dirname in os.listdir(models_dir):
        model_path = os.path.join(models_dir, dirname, "model_weights.pt")
        if os.path.exists(model_path):
            models.append({
                "name": dirname,
                "path": model_path,
                "size_mb": round(os.path.getsize(model_path) / (1024*1024), 2),
                "modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
            })
    
    return {
        "status": "ok",
        "count": len(models),
        "models": models
    }

def check_data_files():
    """Check test data file availability and freshness."""
    data_dir = "../data"
    if not os.path.exists(data_dir):
        return {"status": "error", "message": "Data directory not found"}
    
    data_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith("testing_data_") and filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            data_files.append({
                "name": filename,
                "size_mb": round(os.path.getsize(filepath) / (1024*1024), 2),
                "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            })
    
    return {
        "status": "ok",
        "count": len(data_files),
        "files": data_files
    }

def check_predictions():
    """Check prediction output files."""
    output_dir = "../../src/data"
    predictions_file = os.path.join(output_dir, "latest_predictions.json")
    compact_file = os.path.join(output_dir, "latest_predictions_compact.json")
    
    result = {"status": "ok", "files": {}}
    
    for name, filepath in [("full", predictions_file), ("compact", compact_file)]:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(filepath))
                
                result["files"][name] = {
                    "exists": True,
                    "size_kb": round(os.path.getsize(filepath) / 1024, 2),
                    "age_minutes": round(file_age.total_seconds() / 60, 1),
                    "generated_at": data.get("metadata", {}).get("generated_at"),
                    "successful_models": data.get("metadata", {}).get("summary", {}).get("successful_models", 0),
                    "total_models": data.get("metadata", {}).get("summary", {}).get("total_models_attempted", 0)
                }
                
                # Check if files are too old (more than 24 hours)
                if file_age > timedelta(hours=24):
                    result["files"][name]["warning"] = "File is older than 24 hours"
                    
            except Exception as e:
                result["files"][name] = {
                    "exists": True,
                    "error": f"Failed to parse JSON: {str(e)}"
                }
        else:
            result["files"][name] = {"exists": False}
            result["status"] = "warning"
    
    return result

def check_system_health():
    """Overall system health check."""
    health = {
        "timestamp": datetime.now().isoformat(),
        "models": check_models(),
        "data": check_data_files(),
        "predictions": check_predictions()
    }
    
    # Determine overall status
    statuses = [health["models"]["status"], health["data"]["status"], health["predictions"]["status"]]
    if "error" in statuses:
        health["overall_status"] = "error"
    elif "warning" in statuses:
        health["overall_status"] = "warning"
    else:
        health["overall_status"] = "healthy"
    
    return health

def print_health_report(health, verbose=False):
    """Print a formatted health report."""
    status_colors = {
        "healthy": "🟢",
        "warning": "🟡", 
        "error": "🔴",
        "ok": "✅",
    }
    
    print(f"\n🌟 Solar Flare Prediction System Health Report")
    print(f"📅 Generated: {health['timestamp']}")
    print(f"🎯 Overall Status: {status_colors.get(health['overall_status'], '❓')} {health['overall_status'].upper()}")
    print("=" * 60)
    
    # Models
    models = health["models"]
    print(f"\n📦 Models: {status_colors.get(models['status'], '❓')} {models['status'].upper()}")
    if models["status"] == "ok":
        print(f"   Found {models['count']} model(s)")
        if verbose:
            for model in models["models"]:
                print(f"   - {model['name']} ({model['size_mb']} MB)")
    else:
        print(f"   ❌ {models.get('message', 'Unknown error')}")
    
    # Data
    data = health["data"]
    print(f"\n📊 Data Files: {status_colors.get(data['status'], '❓')} {data['status'].upper()}")
    if data["status"] == "ok":
        print(f"   Found {data['count']} data file(s)")
        if verbose:
            for file in data["files"]:
                print(f"   - {file['name']} ({file['size_mb']} MB)")
    else:
        print(f"   ❌ {data.get('message', 'Unknown error')}")
    
    # Predictions
    predictions = health["predictions"]
    print(f"\n🔮 Predictions: {status_colors.get(predictions['status'], '❓')} {predictions['status'].upper()}")
    for name, info in predictions["files"].items():
        if info["exists"]:
            if "error" in info:
                print(f"   ❌ {name}: {info['error']}")
            else:
                age_str = f"{info['age_minutes']} min ago"
                models_str = f"{info['successful_models']}/{info['total_models']} models"
                print(f"   ✅ {name}: {info['size_kb']} KB, {age_str}, {models_str}")
                if "warning" in info:
                    print(f"      ⚠️  {info['warning']}")
        else:
            print(f"   ❌ {name}: File not found")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Monitor solar flare prediction system health")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Watch mode - repeat every N seconds")
    
    args = parser.parse_args()
    
    def run_check():
        health = check_system_health()
        
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print_health_report(health, args.verbose)
        
        return health["overall_status"]
    
    if args.watch:
        try:
            while True:
                status = run_check()
                if not args.json:
                    print(f"\n⏰ Next check in {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
    else:
        status = run_check()
        # Exit with appropriate code
        exit_codes = {"healthy": 0, "warning": 1, "error": 2}
        exit(exit_codes.get(status, 3))

if __name__ == "__main__":
    main() 