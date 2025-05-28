# How to Collect Ablation Results from Imperial RCS Cluster

## 🎯 **The Issue**

Your ablation jobs are running successfully on the cluster, but the results are being saved to the **cluster filesystem**, not your local machine. The models are saved using the standard EVEREST versioning system in directories like `EVEREST-v[X.X]-M5-72h`.

## 📊 **Current Status**

Based on your logs:
- ✅ **Jobs 1-2**: Completed successfully 
- ✅ **Job 3**: Currently running
- ⏳ **Jobs 4-35**: Queued (will run as resources become available)

Each job takes ~18-19 minutes, so expect **~10.5 hours total** for all 35 experiments.

## 🔍 **Step 1: Check Job Completion Status**

**On the cluster, run:**

```bash
# Check current job status
qstat -u $USER

# Count completed jobs
ls -la everest_component_ablation_fixed.o1174860.* | wc -l

# Check for successful completions
grep -l "completed successfully" everest_component_ablation_fixed.o1174860.*
```

## 📁 **Step 2: Find Ablation Models on Cluster**

**On the cluster, run:**

```bash
# Navigate to project root
cd /rds/general/user/az2221/home/repositories/masters-project

# Find recent EVEREST models (created today)
find . -name "EVEREST-v*-M5-72h" -type d -mtime -1

# List them with timestamps
find . -name "EVEREST-v*-M5-72h" -type d -mtime -1 -exec ls -ld {} \;

# Check how many models were created today
find . -name "EVEREST-v*-M5-72h" -type d -mtime -1 | wc -l
```

## 📋 **Step 3: Create Results Summary on Cluster**

**Create this script on the cluster:**

```bash
# Create collection script
cat > collect_ablation_results.py << 'EOF'
#!/usr/bin/env python3
import os
import json
import glob
from datetime import datetime, timedelta

def find_ablation_models():
    """Find recent M5-72h models that are likely from ablation study."""
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    models = []
    for model_dir in glob.glob("models/EVEREST-v*-M5-72h"):
        if os.path.isdir(model_dir):
            stat = os.stat(model_dir)
            creation_time = datetime.fromtimestamp(stat.st_mtime)
            
            if creation_time > cutoff_time:
                # Try to read metadata
                metadata_path = os.path.join(model_dir, "metadata.json")
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                models.append({
                    'path': model_dir,
                    'name': os.path.basename(model_dir),
                    'created': creation_time.isoformat(),
                    'version': metadata.get('version', 'unknown'),
                    'accuracy': metadata.get('performance', {}).get('accuracy', 'unknown'),
                    'tss': metadata.get('performance', {}).get('TSS', 'unknown')
                })
    
    return sorted(models, key=lambda x: x['created'])

def main():
    print("🔍 Collecting EVEREST ablation results...")
    models = find_ablation_models()
    
    print(f"📁 Found {len(models)} recent M5-72h models:")
    
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model['name']}")
        print(f"    Created: {model['created']}")
        print(f"    Accuracy: {model['accuracy']}")
        print(f"    TSS: {model['tss']}")
        print()
    
    # Save summary
    import csv
    with open('ablation_results_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'path', 'created', 'version', 'accuracy', 'tss'])
        writer.writeheader()
        writer.writerows(models)
    
    print(f"✅ Summary saved to ablation_results_summary.csv")
    print(f"📊 Total experiments completed: {len(models)}/35")

if __name__ == "__main__":
    main()
EOF

# Run the collection script
python collect_ablation_results.py
```

## 📦 **Step 4: Package Results for Download**

**On the cluster, run:**

```bash
# Create results package
mkdir -p ablation_study_results
cd ablation_study_results

# Copy all ablation models
for model in $(find ../models -name "EVEREST-v*-M5-72h" -type d -mtime -1); do
    cp -r "$model" .
done

# Copy logs
cp ../models/ablation/cluster/everest_component_ablation_fixed.o1174860.* .

# Copy summary
cp ../ablation_results_summary.csv .

# Create archive
cd ..
tar -czf ablation_study_results.tar.gz ablation_study_results/

echo "✅ Results packaged in ablation_study_results.tar.gz"
ls -lh ablation_study_results.tar.gz
```

## 💾 **Step 5: Download Results to Local Machine**

**From your local machine, run:**

```bash
# Download the results package
scp az2221@login.hpc.ic.ac.uk:~/repositories/masters-project/ablation_study_results.tar.gz .

# Extract locally
tar -xzf ablation_study_results.tar.gz

# Move to ablation directory
mv ablation_study_results/* models/ablation/trained_models/
```

## 📊 **Step 6: Analyze Results Locally**

**Once downloaded, run:**

```bash
cd models/ablation
python find_ablation_results.py  # Will now find the downloaded models
python analysis.py  # Run your ablation analysis
```

## ⏰ **Timing Expectations**

- **Each job**: ~18-19 minutes
- **Total time**: ~10.5 hours for all 35 jobs
- **Check progress**: Every few hours with `qstat -u $USER`

## 🚨 **If Jobs Fail**

If you see jobs with status 'X' that failed:

```bash
# Check the most recent failed job
ls -t everest_component_ablation_fixed.o1174860.* | head -1 | xargs cat

# Look for error patterns
grep -i "error\|failed\|exception" everest_component_ablation_fixed.o1174860.*
```

## 📞 **Quick Status Check**

**Run this one-liner on the cluster for quick status:**

```bash
echo "Jobs completed: $(ls everest_component_ablation_fixed.o1174860.* 2>/dev/null | wc -l)/35" && echo "Jobs successful: $(grep -l 'completed successfully' everest_component_ablation_fixed.o1174860.* 2>/dev/null | wc -l)" && echo "Models created: $(find models -name 'EVEREST-v*-M5-72h' -mtime -1 | wc -l)"
``` 