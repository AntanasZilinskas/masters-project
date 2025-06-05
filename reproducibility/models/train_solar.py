"""
Script to train EVEREST models and ensure they are saved to models/trained_models
"""

import os
import sys
import importlib
import multiprocessing

# Add parent directory to path so we can import from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create the target directory
os.makedirs("trained_models", exist_ok=True)

# Tell PyTorch to use single process DataLoader
os.environ["PYTORCH_WORKERS"] = "0"

# Import the modules
from everest import RETPlusWrapper
from utils import get_training_data
import numpy as np

def main():
    # --- Configuration ---
    flare_classes = ["C"]         # Only M5 for now, "C", "M", "M5"
    time_windows = ["72"]         # 72-hour window, "24", "48", "72"
    input_shape = (10, 9)
    epochs = 300
    batch_size = 512

    # --- Loop over class × horizon ---
    for flare_class in flare_classes:
        for time_window in time_windows:
            print(f"🚀 Training model for flare class {flare_class} with {time_window}h window")

            # Load & prepare training data
            X_train, y_train = get_training_data(str(time_window), flare_class)

            # Initialize wrapper and train
            model = RETPlusWrapper(input_shape)
            # Modify the wrapper to use 0 workers
            model.train_with_no_workers = True
            model_dir = model.train(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                flare_class=flare_class,
                time_window=time_window
            )

            # Report where everything landed
            print(f"✅ Best weights and metadata stored in: {model_dir}")
            print("-" * 60)

    # Move models if they ended up in the wrong place
    if os.path.exists("models"):
        print("Moving models from models to trained_models...")
        # Use a simple directory move since we're already in the models directory
        for item in os.listdir("models"):
            if item.startswith("EVEREST-v"):
                src = os.path.join("models", item)
                dst = os.path.join("trained_models", item)
                if not os.path.exists(dst):
                    print(f"Moving {item}")
                    os.rename(src, dst)
                else:
                    print(f"Skipping {item} (already exists)")

    print("Training complete. Models should be in trained_models/")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main() 