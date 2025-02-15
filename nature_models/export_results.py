import pandas as pd
import json
import os

# 1) Hardcode the reference data from your images (fill in real numbers):
reference_data = {
    "24": {
        "Recall": [
            ["Huang et al. (Huang et al., 2018)", "—",      "—",      "—"],
            ["Li et al. (Li et al., 2020)",    "—",      "0.817",  "0.889"],
            ["Liu et al. (Liu et al., 2019)",   "0.960",  "0.885",  "0.773"],
            ["Sun et al. (Sun et al., 2022)",   "—",      "0.925",  "0.862"],
            ["Wang et al. (Wang et al., 2020)",  "—",      "0.730",  "0.621"]
        ],
        "Precision": [
            ["Huang et al. (Huang et al., 2018)", "—",     "—",     "—"],
            ["Li et al. (Li et al., 2020)",    "—",     "0.889", "0.906"],
            ["Liu et al. (Liu et al., 2019)",   "0.048", "0.222", "0.541"],
            ["Sun et al. (Sun et al., 2022)",   "—",     "0.595", "0.878"],
            ["Wang et al. (Wang et al., 2020)",  "—",     "0.282", "0.541"]
        ],
        "ACC": [
            ["Huang et al. (Huang et al., 2018)", "—",     "—",     "—"],
            ["Li et al. (Li et., 2020)",    "—",     "0.891", "0.861"],
            ["Liu et al. (Liu et al., 2019)",   "0.921", "0.907", "0.826"],
            ["Sun et al. (Sun et al., 2022)",   "—",     "0.904", "0.879"],
            ["Wang et al. (Wang et al., 2020)",  "—",     "0.945", "0.883"]
        ],
        "BACC": [
            ["Huang et al. (Huang et al., 2018)", "—",     "—",     "—"],
            ["Li et al. (Li et al., 2020)",    "—",     "—",     "—"],
            ["Liu et al. (Liu et al., 2019)",   "0.940", "0.896", "0.806"],
            ["Sun et al. (Sun et al., 2022)",   "—",     "—",     "—"],
            ["Wang et al. (Wang et al., 2020)",  "—",     "—",     "—"]
        ],
        "TSS": [
            ["Huang et al. (Huang et al., 2018)", "—",     "0.662", "0.487"],
            ["Li et al. (Li et al., 2020)",    "—",     "0.749", "0.679"],
            ["Liu et al. (Liu et al., 2019)",   "0.881", "0.792", "0.612"],
            ["Sun et al. (Sun et al., 2022)",   "—",     "0.826", "0.756"],
            ["Wang et al. (Wang et al., 2020)",  "—",     "0.681", "0.553"]
        ]
    },
    "48": {
        "Recall": [
            ["Huang et al. (Huang et al., 2018)", "—",    "—",    "—"],
            ["Li et al. (Li et al., 2020)",    "—",    "0.850", "0.900"],
            ["Liu et al. (Liu et al., 2019)",   "0.960", "0.885", "0.780"],
            ["Sun et al. (Sun et al., 2022)",   "—",    "0.910", "0.850"],
            ["Wang et al. (Wang et al., 2020)",  "—",    "0.720", "0.610"]
        ],
        "Precision": [
            # ... fill in as needed
        ],
        "ACC": [
            # ... fill in as needed
        ],
        "BACC": [
            # ... fill in as needed
        ],
        "TSS": [
            # ... fill in as needed
        ]
    },
    "72": {
        "Recall": [
            ["Huang et al. (Huang et al., 2018)", "—",    "—",    "—"],
            ["Li et al. (Li et al., 2020)",    "—",    "0.810", "0.860"],
            ["Liu et al. (Liu et al., 2019)",   "0.940", "0.860", "0.750"],
            ["Sun et al. (Sun et al., 2022)",   "—",    "0.900", "0.840"],
            ["Wang et al. (Wang et al., 2020)",  "—",    "0.700", "0.600"]
        ],
        "Precision": [
            # ... fill in as needed
        ],
        "ACC": [
            # ... fill in as needed
        ],
        "BACC": [
            # ... fill in as needed
        ],
        "TSS": [
            # ... fill in as needed
        ]
    }
}

# 2) Load your experimental "This work" results from JSON
results_file = "this_work_results.json"
if os.path.exists(results_file):
    with open(results_file, "r") as f:
        all_test_results = json.load(f)
else:
    all_test_results = {}

# Debug print: show loaded JSON keys and structure
print("Loaded JSON keys:", list(all_test_results.keys()))
if all_test_results:
    for horizon, data in all_test_results.items():
        print(f"Time window {horizon} contains flare classes:", list(data.keys()))

# Define mapping from export metric names to JSON keys.
metric_key_map = {
    "Recall": "recall",
    "Precision": "precision",
    "ACC": "accuracy",
    "BACC": "balanced_accuracy",
    "TSS": "TSS"
}

# Helper to fetch a metric from the loaded JSON results or return "N/A"
def get_this_work_metric(horizon, flare_class, metric_key):
    return str(
        all_test_results.get(horizon, {})
                       .get(flare_class, {})
                       .get(metric_key, "N/A")
    )

# 3) Hardcode the "Nature paper" values at each horizon (from the state-of-the-art table).
nature_paper = {
    "24": {
        "Recall":    ["0.853", "0.842", "0.891"],
        "Precision": ["0.977", "0.848", "0.949"],
        "ACC":       ["0.964", "0.928", "0.915"],
        "BACC":      ["0.926", "0.919", "0.917"],
        "TSS":       ["0.818", "0.839", "0.835"]
    },
    "48": {
        "Recall":    ["0.739", "0.735", "0.722"],
        "Precision": ["0.890", "0.823", "0.812"],
        "ACC":       ["0.923", "0.907", "0.896"],
        "BACC":      ["0.864", "0.857", "0.848"],
        "TSS":       ["0.736", "0.728", "0.719"]
    },
    "72": {
        "Recall":    ["0.717", "0.708", "0.702"],
        "Precision": ["0.872", "0.812", "0.809"],
        "ACC":       ["0.906", "0.883", "0.863"],
        "BACC":      ["0.856", "0.843", "0.834"],
        "TSS":       ["0.729", "0.714", "0.709"]
    }
}

# 4) Build one big table: for each horizon and metric, add:
#    A) the reference rows, B) a row for "Nature paper", and C) a row for "This work" (from JSON).
all_rows = []
metrics = ["Recall", "Precision", "ACC", "BACC", "TSS"]
horizons = ["24", "48", "72"]

for horizon in horizons:
    for metric in metrics:
        # A) Add the reference state-of-the-art rows (from reference_data)
        if horizon in reference_data and metric in reference_data[horizon]:
            for ref_row in reference_data[horizon][metric]:
                method_name = ref_row[0]
                val_m5      = ref_row[1]
                val_m       = ref_row[2]
                val_c       = ref_row[3]
                all_rows.append([horizon, metric, method_name, val_m5, val_m, val_c])
        
        # B) Add the "Nature paper" row for this horizon+metric
        if horizon in nature_paper and metric in nature_paper[horizon]:
            np_vals = nature_paper[horizon][metric]  # [M5, M, C]
            all_rows.append([horizon, metric, "Nature paper",
                             np_vals[0], np_vals[1], np_vals[2]])
        
        # C) Add the "This work" row from JSON
        key = metric_key_map[metric]  # e.g., "ACC" -> "accuracy"
        val_m5 = get_this_work_metric(horizon, "M5", key)
        val_m  = get_this_work_metric(horizon, "M",  key)
        val_c  = get_this_work_metric(horizon, "C",  key)
        all_rows.append([horizon, metric, "This work", val_m5, val_m, val_c])

# 5) Convert to DataFrame and export to Excel and text.
df = pd.DataFrame(all_rows, columns=["Horizon", "Metric", "Method", "≥ M5.0 class", "≥ M class", "≥ C class"])

excel_filename = "results_all_horizons.xlsx"
df.to_excel(excel_filename, index=False)
print(f"Excel file saved as {excel_filename}")

txt_filename = "results_all_horizons.txt"
with open(txt_filename, "w") as f:
    f.write(df.to_string(index=False))
print(f"Text file saved as {txt_filename}")