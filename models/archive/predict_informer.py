import logging
import torch
import numpy as np
import matplotlib.pyplot as plt  # <-- new import for plotting
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pprint

# 1) Local import of the same-folder module (no leading dot).
from informer import (
    select_device,   # includes MPS checking
    GOESDataset,
    Informer,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


class GOESDataset(Dataset):
    """
    Loads GOES netCDF files containing "avg1m_g13" in the filename (e.g. sci_xrsf-l2-avg1m_g16_d20180908_v2-2-0.nc),
    merges them, and creates sliding windows (lookback -> forecast).
    """

    def __init__(self,
                 data_dir,
                 lookback_len=24,
                 forecast_len=24,
                 step_per_hour=60,
                 train=True,
                 train_split=0.8,
                 max_files=None):
        super().__init__()
        self.lookback_len = lookback_len * step_per_hour
        self.forecast_len = forecast_len * step_per_hour
        self.train = train

        # 1) Debugging: Show pattern and matched files.
        pattern = os.path.join(data_dir, "*avg1m_g13*.nc")
        print("\n[DEBUG] GOESDataset __init__ (predict_informer.py)")
        print(f"[DEBUG] data_dir = {data_dir}")
        print(f"[DEBUG] Glob pattern = {pattern}")
        all_files = sorted(glob.glob(pattern))
        print("[DEBUG] All matching files found:")
        pprint.pprint(all_files)

        if max_files is not None:
            all_files = all_files[:max_files]
            print(f"[DEBUG] After max_files={max_files}, truncated list:")
            pprint.pprint(all_files)

        logging.info(
            f"Found {len(all_files)} netCDF files in '{data_dir}' with 'avg1m_g13'")
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No .nc files matching '*avg1m_g13*.nc' in {data_dir}"
            )

        # ------------------------------------------------------------------------------
        # 2) Load flux data from each file
        # ------------------------------------------------------------------------------
        flux_list = []
        for fpath in all_files:
            print(f"[DEBUG] Attempting to open: {fpath}")
            try:
                ds = xr.open_dataset(fpath)
                if 'xrsb_flux' in ds.variables:
                    flux_var = 'xrsb_flux'
                elif 'b_flux' in ds.variables:
                    flux_var = 'b_flux'
                elif 'a_flux' in ds.variables:
                    flux_var = 'a_flux'
                else:
                    ds.close()
                    logging.warning(
                        f"No recognized flux variable in {fpath}, skipping.")
                    continue

                flux_vals = ds[flux_var].values
                ds.close()
                flux_list.append(flux_vals)
                print(
                    f"[DEBUG] Loaded {len(flux_vals)} timesteps from {fpath} (var='{flux_var}')")
            except Exception as e:
                logging.warning(f"Could not load {fpath}. Error: {e}")
                continue

        if len(flux_list) == 0:
            raise ValueError(
                "No valid flux data found among the selected netCDF files.")

        # Concatenate all flux arrays into a single time-series
        all_flux = np.concatenate(flux_list, axis=0)

        # 3) Fill NaNs and apply small transform if desired
        all_flux = np.nan_to_num(all_flux, nan=1e-9)
        self.data = np.log1p(all_flux)  # e.g. log-transform
        print(f"[DEBUG] Total concatenated timesteps = {len(self.data)}")

        # 4) Train/test split
        N = len(self.data)
        split_index = int(N * train_split)
        if train:
            self.data = self.data[:split_index]
            logging.info(f"Training portion: {len(self.data)} samples")
        else:
            self.data = self.data[split_index:]
            logging.info(
                f"Validation/Testing portion: {len(self.data)} samples")

        # 5) Build sliding window indices
        self.indices = []
        # Original code gave range(max_start), which is empty if max_start==0
        max_start = len(self.data) - self.lookback_len - self.forecast_len
        # Add +1 so that if max_start==0, we get exactly one valid window
        for i in range(max_start + 1 if max_start >= 0 else 0):
            self.indices.append(i)

        logging.debug(f"Total sliding-window samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.lookback_len
        x_seq = self.data[start:end]  # shape: [lookback_len]
        y_seq = self.data[end:end + self.forecast_len]  # shape: [forecast_len]

        import torch
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)
        return x_tensor, y_tensor


def load_trained_informer(model_path, device='cpu',
                          lookback_len=24,
                          forecast_len=24,
                          d_model=64,
                          n_heads=4,
                          d_ff=128,
                          enc_layers=2,
                          dec_layers=1,
                          dropout=0.1):
    """
    Reconstructs the Informer model architecture, then loads its saved state dict
    from the specified checkpoint (model_path).
    """
    model = Informer(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dropout=dropout,
        lookback_len=lookback_len,
        forecast_len=forecast_len
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Loaded Informer model weights from {model_path}")
    return model


def predict_and_compare(model, test_dataset, device='cpu'):
    """
    Runs inference on the test dataset to get 24-hour predictions from each
    24-hour input window, then compares them with the actual target data.

    Returns the mean squared error over the entire test set, along with
    all predictions/targets for optional further analysis or plotting.
    """
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_predictions = []
    all_targets = []
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)  # shape: [1, lookback_len]
            y = y.to(device)  # shape: [1, forecast_len]

            # Pass zero placeholder if your model expects (src, tgt) in
            # forward()
            future_dummy = torch.zeros_like(y).to(device)

            # The model forward pass
            pred = model(x, future_dummy)  # shape: [1, forecast_len]

            # Collect as NumPy arrays
            pred_np = pred.cpu().numpy().flatten()
            y_np = y.cpu().numpy().flatten()

            all_predictions.append(pred_np)
            all_targets.append(y_np)

            # Compute MSE for this window
            mse = np.mean((pred_np - y_np)**2)
            mse_sum += mse
            count += 1

            if (batch_idx + 1) % 50 == 0:
                logging.debug(
                    f"[{batch_idx+1}/{len(test_loader)}] Sample MSE: {mse:.6f}")

    overall_mse = mse_sum / max(count, 1)
    logging.info(f"Prediction MSE on test dataset: {overall_mse:.6f}")

    return all_predictions, all_targets, overall_mse


def plot_predictions(all_predictions, all_targets, num_windows=3):
    """
    Plots predicted vs. actual flux for a few windows from the test dataset.
    """
    num_windows = min(num_windows, len(all_predictions))
    if num_windows == 0:
        logging.warning("No windows to plot (all_predictions is empty).")
        return

    fig, axes = plt.subplots(
        num_windows, 1, figsize=(
            8, 4 * num_windows), sharex=False)

    if num_windows == 1:
        # If there's only one window, axes is not a list
        axes = [axes]

    for i in range(num_windows):
        ax = axes[i]
        pred = all_predictions[i]
        tgt = all_targets[i]
        ax.plot(pred, label="Predicted")
        ax.plot(tgt, label="Actual")
        ax.set_title(f"Test Window {i+1}")
        ax.set_xlabel("Forecast timestep")
        ax.set_ylabel("Flux (transformed)")  # or the appropriate unit
        ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    MODEL_PATH = "informer-24h-new_test_split.pth"
    DATA_DIR = (
        "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/"
        "Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"
    )

    # Select MPS / CUDA / CPU
    device = select_device()
    logging.info(f"Using device: {device}")

    # 1) Load the previously trained Model
    model = load_trained_informer(
        model_path=MODEL_PATH,
        device=device,
        lookback_len=24,
        forecast_len=24,
        d_model=64,
        n_heads=4,
        d_ff=128,
        enc_layers=2,
        dec_layers=1,
        dropout=0.1
    )

    # 2) Construct the test dataset from exactly 2 adjacent daily files
    test_dataset = GOESDataset(
        data_dir=DATA_DIR,
        lookback_len=24,  # hours
        forecast_len=24,  # hours
        step_per_hour=60,  # 1-min data => 60 steps/hour => 1440 steps/day
        train=False,
        train_split=0.0,  # all data loaded goes to the "test" split
        max_files=2       # load exactly 2 daily files => 2880 steps total
    )

    # 3) Predict & Compare
    all_preds, all_tgts, test_mse = predict_and_compare(
        model, test_dataset, device=device)

    # 4) Log a snippet of results
    logging.info("Example predictions vs. targets (first 3 windows):")
    for i in range(min(3, len(all_preds))):
        logging.info(
            f"Window {i+1} - Pred: {all_preds[i][:5]}..., Tgt: {all_tgts[i][:5]}...")

    # 5) Plot a few windows
    plot_predictions(all_preds, all_tgts, num_windows=3)


if __name__ == "__main__":
    main()
