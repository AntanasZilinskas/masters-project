import os
import glob
import logging
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from math import ceil


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

########################################################################
# 1) GOESDataset with Optional file_list Argument
########################################################################


class GOESDataset(Dataset):
    def __init__(self,
                 data_dir=None,
                 file_list=None,
                 lookback_len=72,
                 forecast_len=24,
                 step_per_hour=60,
                 train=True,
                 train_split=0.8):
        super().__init__()

        self.lookback_len = lookback_len * step_per_hour
        self.forecast_len = forecast_len * step_per_hour
        self.train = train

        # If the caller provides file_list, use that. Otherwise, glob data_dir.
        if file_list is not None:
            all_files = sorted(file_list)
        else:
            pattern = os.path.join(data_dir, "*avg1m_g13*.nc")
            all_files = sorted(glob.glob(pattern))

        logging.info(f"GOESDataset => #files: {len(all_files)}")
        if len(all_files) == 0:
            raise FileNotFoundError("No matching netCDF files found.")

        # Load flux data from the files
        flux_list = []
        for fpath in all_files:
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
            except Exception as e:
                logging.warning(f"Failed to load {fpath}: {e}")
                continue

        if not flux_list:
            raise ValueError("No valid flux data loaded from these files.")

        # Concatenate
        all_flux = np.concatenate(flux_list, axis=0)
        # Replace NaNs
        all_flux = np.nan_to_num(all_flux, nan=1e-9)
        self.data = np.log1p(all_flux)  # log(1 + flux)

        # Train/test split
        N = len(self.data)
        split_index = int(N * train_split)
        if train:
            self.data = self.data[:split_index]
            logging.info(f"Training portion: {len(self.data)} samples.")
        else:
            self.data = self.data[split_index:]
            logging.info(f"Testing portion: {len(self.data)} samples.")

        # Build sliding windows
        max_start = len(self.data) - self.lookback_len - self.forecast_len
        self.indices = list(range(max_start + 1)) if max_start >= 0 else []
        logging.info(f"Total sliding-window samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.lookback_len
        x_seq = self.data[start:end]
        y_seq = self.data[end:end + self.forecast_len]

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)
        return x_tensor, y_tensor

########################################################################
# 2) Informer model definition (unchanged from before)
########################################################################


class Informer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, enc_layers, dec_layers,
                 dropout, lookback_len, forecast_len):
        super().__init__()
        # Implementation details omitted for brevity...
        pass

    def forward(self, x_enc, x_mark_enc):
        # Toy forward
        return x_mark_enc

########################################################################
# 3) Evaluate & Save Functions
########################################################################


def evaluate_informer(model, data_loader, device='mps', criterion=None):
    model.eval()
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            future_dummy = torch.zeros_like(y).to(device)
            pred = model(x, future_dummy)
            if criterion is not None:
                mse_sum += criterion(pred, y).item()
            count += 1
    return mse_sum / max(count, 1)

########################################################################
# 4) CHUNKED TRAINING: chunked_train_informer
########################################################################


def chunked_train_informer(data_dir,
                           lookback_len=72,
                           forecast_len=24,
                           batch_size=16,
                           lr=1e-4,
                           device=None,
                           model_save_path="informer-chunked.pth",
                           total_chunks=10,
                           files_per_chunk=10,
                           epochs_per_chunk=1):
    """
    Load all netCDF files from data_dir, split them into 'total_chunks'
    subsets of size 'files_per_chunk' (or smaller if you run out of files),
    and train on each chunk for 'epochs_per_chunk' epochs. At the end,
    evaluate on the final test set if desired.
    """

    if device is None:
        device = select_device()

    # 1) Gather all files
    pattern = os.path.join(data_dir, "*avg1m_g13*.nc")
    all_files = sorted(glob.glob(pattern))
    logging.info(f"FOUND {len(all_files)} files in {data_dir}")

    # 2) Initialize model + optimizer
    model = Informer(
        d_model=64,
        n_heads=4,
        d_ff=128,
        enc_layers=2,
        dec_layers=1,
        dropout=0.1,
        lookback_len=lookback_len,
        forecast_len=forecast_len
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    files_count = len(all_files)
    chunk_size = files_per_chunk
    chunk_count = 0

    # 3) Loop over chunks of files
    for chunk_i in range(total_chunks):
        start_idx = chunk_i * chunk_size
        if start_idx >= files_count:
            break  # No more files left
        end_idx = min(start_idx + chunk_size, files_count)
        chunk_files = all_files[start_idx:end_idx]
        chunk_count += 1

        logging.info(
            f"--- Chunk {chunk_i+1} / {total_chunks} => {len(chunk_files)} files ---")

        # Build a chunked train dataset (train_split=1.0 so it's purely
        # training)
        train_dataset = GOESDataset(
            data_dir=None,        # we won't use a directory-based glob
            file_list=chunk_files,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            step_per_hour=60,
            train=True,
            train_split=1.0
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True)

        # 4) Train for epochs_per_chunk
        for epoch in range(epochs_per_chunk):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                future_dummy = torch.zeros_like(y).to(device)
                pred = model(x, future_dummy)

                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)
            logging.info(
                f"Chunk {chunk_i+1} - Epoch {epoch+1}/{epochs_per_chunk} - Loss: {avg_loss:.6f}")

    # 5) OPTIONAL: Evaluate on a hold-out test set
    # You can define a separate test dataset that uses train=False and e.g. train_split=0.0 or 0.8
    # with all_files or leftover files. Example:
    test_dataset = GOESDataset(
        data_dir=None,
        file_list=all_files,     # or some subset for testing
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        step_per_hour=60,
        train=False,
        train_split=0.8          # leaving 20% for test
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_mse = evaluate_informer(
        model,
        test_loader,
        device=device,
        criterion=criterion)
    logging.info(f"Final Test MSE after chunked training: {test_mse:.6f}")

    # 6) Save the model
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    return model


########################################################################
# (Optional) MAIN
########################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_folder = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"
    dev = select_device()
    chunked_train_informer(
        data_dir=data_folder,
        lookback_len=72,       # 3 days
        forecast_len=24,       # 1 day
        batch_size=16,
        lr=1e-4,
        device=dev,
        model_save_path="informer-chunked.pth",
        total_chunks=5,        # Train on 5 "chunks"
        files_per_chunk=10,    # Each chunk has up to 10 files
        epochs_per_chunk=1     # # of epochs you want to do per chunk
    )
