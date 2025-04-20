import os
import glob
import logging
import pprint
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.nn.utils as utils
import torch.nn.functional as F
import datetime

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

#######################################################################
# 1) DEVICE SELECTION (MPS/CUDA/CPU)
#######################################################################


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

#######################################################################
# 2) DATASET (For NetCDF files)
#######################################################################


class GOESDataset(Dataset):
    """
    Loads GOES netCDF files containing 'avg1m_g13' in the filename,
    merges them, and creates sliding windows (lookback -> forecast).
    """

    def __init__(self,
                 data_dir,
                 lookback_len=72,      # default: 3 days
                 forecast_len=24,      # default: 1 day
                 step_per_hour=60,
                 train=True,
                 train_split=0.8,
                 max_files=None):
        super().__init__()
        self.lookback_len = lookback_len * step_per_hour
        self.forecast_len = forecast_len * step_per_hour
        self.train = train

        # ----------------------------------------------------------------------
        # 1) Print debugging for pattern & file discovery
        # ----------------------------------------------------------------------
        pattern = os.path.join(data_dir, "*avg1m_g13*.nc")
        print("\n[DEBUG] GOESDataset __init__")
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
                f"No .nc files matching '*avg1m_g13*.nc' in {data_dir}")

        # ----------------------------------------------------------------------
        # 2) Load flux data (debugging each file)
        # ----------------------------------------------------------------------
        flux_list = []
        for fpath in all_files:
            print(f"[DEBUG] Attempting to open: {fpath}")
            try:
                ds = xr.open_dataset(fpath)
                # Check each possible flux variable
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
            except Exception as err:
                logging.warning(
                    f"Could not load {fpath}, skipping. Error: {err}")
                continue

        if not flux_list:
            raise ValueError(
                "No valid flux data found in selected netCDF files.")

        all_flux = np.concatenate(flux_list, axis=0)
        # Do NOT replace NaNs; we want to disregard samples containing null values.
        # Clip values to avoid log1p(x) for x < -1 which returns nan
        all_flux = np.clip(all_flux, a_min=-0.999, a_max=None)
        self.data = np.log1p(all_flux)
        print(f"[DEBUG] Total concatenated timesteps = {len(self.data)}")

        # ----------------------------------------------------------------------
        # 3) Train/Test split
        # ----------------------------------------------------------------------
        N = len(self.data)
        split_index = int(N * train_split)
        if train:
            self.data = self.data[:split_index]
            logging.info(f"Training portion: {len(self.data)} samples")
        else:
            self.data = self.data[split_index:]
            logging.info(
                f"Validation/Testing portion: {len(self.data)} samples")

        # ----------------------------------------------------------------------
        # 4) Build sliding-window indices (+1 fix)
        # ----------------------------------------------------------------------
        max_start = len(self.data) - self.lookback_len - self.forecast_len
        self.indices = []
        for i in range(max_start + 1 if max_start >= 0 else 0):
            self.indices.append(i)

        logging.info(f"Total sliding-window samples: {len(self.indices)}")
        window_length = self.lookback_len + self.forecast_len
        valid_indices = [i for i in self.indices if not np.isnan(
            self.data[i:i + window_length]).any()]
        logging.info(
            f"Filtered sliding-window samples (without NaN): {len(valid_indices)} out of {len(self.indices)}")
        self.indices = valid_indices

        # ----------------------------------------------------------------------
        # 5) Standard-scale the data
        # ----------------------------------------------------------------------
        train_mean = np.nanmean(self.data)
        train_std = np.nanstd(self.data)
        self.data = (self.data - train_mean) / (train_std + 1e-8)

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

#######################################################################
# 2B) DATASET (Using the precombined Parquet file)
#######################################################################


class GOESParquetDataset(Dataset):
    """
    Loads the precombined Parquet file (created by data processing) that
    contains minute-level 'flux' data. It then builds sliding-windows for
    forecasting.
    """

    def __init__(self,
                 parquet_file,
                 lookback_len=72,      # default: 3 days
                 forecast_len=24,      # default: 1 day
                 train=True,
                 train_split=0.8):
        super().__init__()
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError(
                "Parquet file must contain 'time' and 'flux' columns.")
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)

        # Assuming the parquet file is continuous minute data with no gaps because
        # the processing ensured that for each minute some satellite data was
        # used.
        self.data = df['flux'].values.astype(np.float32)
        # Do NOT replace NaNs; we want to disregard samples containing null values.
        # Clip values to avoid np.log1p(x) with x < -1 which returns nan
        self.data = np.clip(self.data, a_min=-0.999, a_max=None)
        self.data = np.log1p(self.data)

        # Save lengths in number of time-steps (minutes)
        # lookback_len is given in hours here (72 h = 3 days)
        self.lookback_len = lookback_len * 60
        # forecast_len in hours (24 h = 1 day)
        self.forecast_len = forecast_len * 60
        self.train = train

        total_steps = len(self.data)
        split_index = int(total_steps * train_split)
        if train:
            self.data = self.data[:split_index]
            logging.info(
                f"[ParquetDataset] Training portion: {len(self.data)} samples")
        else:
            self.data = self.data[split_index:]
            logging.info(
                f"[ParquetDataset] Validation/Testing portion: {len(self.data)} samples")

        max_start = len(self.data) - self.lookback_len - self.forecast_len
        self.indices = list(range(max_start + 1 if max_start >= 0 else 0))
        logging.info(
            f"[ParquetDataset] Total sliding-window samples: {len(self.indices)}")
        window_length = self.lookback_len + self.forecast_len
        valid_indices = [i for i in self.indices if not np.isnan(
            self.data[i:i + window_length]).any()]
        logging.info(
            f"[ParquetDataset] Filtered sliding-window samples (without NaN): {len(valid_indices)} out of {len(self.indices)}")
        self.indices = valid_indices

        # ----------------------------------------------------------------------
        # 5) Standard-scale the data and store normalization parameters
        # ----------------------------------------------------------------------
        train_mean = np.nanmean(self.data)
        train_std = np.nanstd(self.data)
        self.data = (self.data - train_mean) / (train_std + 1e-8)
        self.mean = train_mean
        self.std = train_std

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

#######################################################################
# 3) INFORMER MODEL (MINIMAL VERSION)
#######################################################################


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generate a causal mask for the decoder so positions cannot attend
    to subsequent positions. Shape: [sz, sz]
    """
    # Create an upper triangular matrix (with diagonal offset=1) so that
    # positions j > i are masked with -inf, and the allowed positions are 0.
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class TokenEmbedding(nn.Module):
    """
    Embeds the input sequence from (batch_size, 1, seq_len)
    into (batch_size, d_model, seq_len).
    Uses zero padding instead of circular to avoid wrap-around.
    """

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='zeros'
        )
        # Comment out or remove BN:
        # self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        out = self.tokenConv(x)
        # out = self.bn(out)   # Remove BN
        return out


class PositionalEmbedding(nn.Module):
    """
    Fixed positional encoding for up to max_len positions.
    """

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model] (after permutation if needed)
        Returns a slice of the positional embeddings with shape matching x.
        """
        seq_len = x.size(1)
        # self.pe is [1, max_len, d_model], so we slice to [1, seq_len,
        # d_model]
        return self.pe[:, :seq_len, :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x2, _ = self.slf_attn(x, x, x,
                              attn_mask=None,
                              key_padding_mask=None)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(x2)
        x = self.norm2(x)
        return x


class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None):
        """
        x: decoder input [tgt_len, batch_size, d_model]
        enc_out: [src_len, batch_size, d_model]
        tgt_mask: causal mask for self-attn
        """
        x2, _ = self.slf_attn(x, x, x,
                              attn_mask=tgt_mask,
                              key_padding_mask=None)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2, _ = self.cross_attn(x, enc_out, enc_out)
        x = x + self.dropout(x2)
        x = self.norm2(x)

        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(x2)
        x = self.norm3(x)
        return x


class InformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask)
        return x


class Informer(nn.Module):
    """
    Minimal Informer-like structure for univariate time-series forecasting.
    Usually applies:
      - token embedding (conv + BN)
      - positional embedding
      - multiple encoder layers
      - multiple decoder layers
      - final projection
    """

    def __init__(self,
                 d_model=128,
                 n_heads=8,
                 d_ff=256,
                 enc_layers=3,
                 dec_layers=2,
                 dropout=0.1,
                 lookback_len=72,
                 forecast_len=24):
        super().__init__()
        if torch.backends.mps.is_available():
            dropout = 0.0
        self.lookback_len = lookback_len
        self.forecast_len = forecast_len

        self.token_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        self.downsample = nn.Conv1d(in_channels=d_model,
                                    out_channels=d_model,
                                    kernel_size=4,
                                    stride=4)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.encoder = InformerEncoder(
            d_model, n_heads, d_ff, enc_layers, dropout)
        self.decoder = InformerDecoder(
            d_model, n_heads, d_ff, dec_layers, dropout)

        # Bridging layer to transform the encoder's last output for the
        # decoder.
        self.bridge = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt, use_causal_mask=True, tgt_mask=None):
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape

        # Encoder: embed, downsample, and add positional encoding.
        enc_in = src.unsqueeze(1)  # [batch_size, 1, src_len]
        enc_in = self.token_embedding(enc_in)  # [batch_size, d_model, src_len]
        enc_in = self.downsample(enc_in)  # [batch_size, d_model, new_src_len]
        enc_in = enc_in.permute(2, 0, 1)  # [new_src_len, batch_size, d_model]

        # [batch_size, new_src_len, d_model]
        enc_in_pe = enc_in.permute(1, 0, 2)
        enc_in_pe = enc_in_pe + self.pos_embedding(enc_in_pe)
        # [new_src_len, batch_size, d_model]
        enc_in = enc_in_pe.permute(1, 0, 2)

        enc_out = self.encoder(enc_in)  # [new_src_len, batch_size, d_model]
        # Bridge: transform the final encoder output (last time step)
        # to serve as an initializer for the decoder.
        # [1, batch_size, d_model]
        bridge_out = self.bridge(enc_out[-1]).unsqueeze(0)

        # Decoder: embed target sequence and add positional encoding.
        dec_in = tgt.unsqueeze(1)  # [batch_size, 1, tgt_len]
        dec_in = self.token_embedding(dec_in)  # [batch_size, d_model, tgt_len]
        dec_in = dec_in.permute(2, 0, 1)  # [tgt_len, batch_size, d_model]

        dec_in_pe = dec_in.permute(1, 0, 2)  # [batch_size, tgt_len, d_model]
        dec_in_pe = dec_in_pe + self.pos_embedding(dec_in_pe)
        dec_in = dec_in_pe.permute(1, 0, 2)  # [tgt_len, batch_size, d_model]
        # Add the bridged encoder summary to the decoder input
        dec_in = dec_in + bridge_out

        if use_causal_mask and tgt_mask is None:
            tgt_mask = generate_square_subsequent_mask(tgt_len).to(src.device)

        dec_out = self.decoder(dec_in, enc_out, tgt_mask=tgt_mask)

        dec_out = self.proj(dec_out)
        dec_out = dec_out.permute(1, 0, 2).squeeze(-1)

        return dec_out

#######################################################################
# 4) TRAINING
#######################################################################


def train_informer(data_source,
                   lookback_len=24,     # For example, 24 hours
                   forecast_len=12,     # For example, 12 hours
                   epochs=10,
                   batch_size=8,
                   lr=1e-6,
                   device=None,
                   parquet_file=None,
                   data_dir=None,
                   max_files=None,
                   model_save_path="informer-archive.pth",
                   early_stopping_patience=3,
                   checkpoint_every=2,
                   max_train_samples=50000,
                   max_val_samples=10000):
    """
    Extended version of train_informer that creates a timestamped run folder and
    saves all models (checkpoints, best, and final) in that folder.
    It also uses increased training/validation samples and a larger model.
    """

    # Create a timestamped run folder within "model_archives"
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join("model_archives", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Run directory: {run_dir}")

    if device is None:
        device = select_device()
    logging.info(f"Using device: {device}")

    # --------------------------------------------------------------------------
    # 1) Prepare datasets
    # --------------------------------------------------------------------------
    if parquet_file is not None:
        logging.info(f"Using Parquet file: {parquet_file}")
        train_dataset = GOESParquetDataset(
            parquet_file=parquet_file,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            train=True,
            train_split=0.8
        )
        val_dataset = GOESParquetDataset(
            parquet_file=parquet_file,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            train=False,
            train_split=0.8
        )
    else:
        logging.info(f"Loading netCDF files from: {data_dir}")
        train_dataset = GOESDataset(
            data_dir=data_dir,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            step_per_hour=60,
            train=True,
            train_split=0.8,
            max_files=max_files
        )
        val_dataset = GOESDataset(
            data_dir=data_dir,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            step_per_hour=60,
            train=False,
            train_split=0.8,
            max_files=max_files
        )

    # ---- NEW: Truncate the data for a quicker test run ----
    # (max_train_samples / max_val_samples can be adjusted)
    # This directly slices the underlying .data array in each dataset.
    if max_train_samples is not None and max_train_samples < len(
            train_dataset.indices):
        train_dataset.indices = train_dataset.indices[:max_train_samples]
        logging.info(
            f"Truncating train dataset to {max_train_samples} samples.")

    if max_val_samples is not None and max_val_samples < len(
            val_dataset.indices):
        val_dataset.indices = val_dataset.indices[:max_val_samples]
        logging.info(
            f"Truncating validation dataset to {max_val_samples} samples.")

    logging.info(f"Final train samples: {len(train_dataset.indices)}")
    logging.info(f"Final validation samples: {len(val_dataset.indices)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,      # Enables faster CPU->GPU transfer
        # Parallel data loading (adjust number based on your CPU)
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    # After you create train_dataset (and before DataLoader), do something
    # like:
    all_x = []
    for idx in train_dataset.indices[:200]:  # check first 200 samples
        x, y = train_dataset[idx]
        all_x.append(x.numpy())

    # Convert to a single array
    all_x = np.concatenate(all_x)
    print("Min value in training samples:", np.min(all_x))
    print("Max value in training samples:", np.max(all_x))
    print(
        "Any NaN in training samples?",
        np.isnan(all_x).any(),
        np.isnan(all_x).sum())
    print("Any Inf in training samples?", np.isinf(all_x).any())
    # Save normalization parameters for deployment.
    train_mean = np.nanmean(all_x)
    train_std = np.nanstd(all_x)
    np.save("scaling_params.npy", {"mean": train_mean, "std": train_std})

    # --------------------------------------------------------------------------
    # 2) Build the model
    # --------------------------------------------------------------------------
    model = Informer(
        d_model=128,
        n_heads=8,
        d_ff=256,
        enc_layers=3,
        dec_layers=2,
        dropout=0.1,
        lookback_len=lookback_len,
        forecast_len=forecast_len
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9)

    # --------------------------------------------------------------------------
    # 3) Training Setup (Early Stopping + Checkpoints)
    # --------------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_counter = 0

    # For optional logging/plotting, track losses per epoch
    train_losses = []
    val_losses = []

    # --------------------------------------------------------------------------
    # 4) Training Loop
    # --------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Use tqdm progress bar for training
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch") as tepoch:
            for batch_idx, (x, y) in enumerate(tepoch):
                # Move batch data to device.
                x, y = x.to(device), y.to(device)
                batch_size = x.size(0)

                # Determine the current teacher forcing ratio.
                # For example, start with ratio=1.0 and decay linearly to 0.0
                # over all epochs.
                # Adjust the decay constant as needed
                teacher_forcing_ratio = np.exp(-epoch / (epochs / 5))

                # Use teacher forcing or autoregressive decoding based on a
                # random draw.
                if torch.rand(batch_size).mean(
                ).item() < teacher_forcing_ratio:
                    # Teacher Forcing Mode: use the ground truth forecast as
                    # decoder input.
                    dec_input = y
                else:
                    # Autoregressive Mode: initialize decoder input with the
                    # last observed value.
                    forecast_steps = y.shape[1]
                    dec_input = x[:, -1:].clone()   # shape: [batch_size, 1]
                    # Define a temperature parameter to control sampling
                    # randomness.
                    temperature = 0.7
                    # Iterate forecast_steps - 1 times so that final sequence
                    # length equals forecast_steps.
                    for t in range(forecast_steps - 1):
                        current_length = dec_input.shape[1]
                        # Create a dummy sequence with length (current_length +
                        # 1).
                        dummy = torch.zeros(
                            batch_size, current_length + 1, device=device)
                        dummy[:, :current_length] = dec_input
                        with torch.no_grad():
                            pred_full = model(
                                x, dummy)  # shape: [batch_size, current_length+1]
                        # shape: [batch_size, 1]
                        pred_next = pred_full[:, -1].unsqueeze(1)
                        # Add Gaussian noise scaled by temperature as a soft
                        # sampling mechanism.
                        noise = torch.randn_like(pred_next) * temperature
                        pred_next = pred_next + noise
                        dec_input = torch.cat([dec_input, pred_next], dim=1)

                # Run the forward pass to compute the batch prediction.
                pred = model(x, dec_input)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(avg_train_loss)

        # ----------------------------------------------------------------------
        # Evaluate on the validation set
        # ----------------------------------------------------------------------
        val_mse = evaluate_informer(
            model,
            val_loader,
            device=device,
            criterion=criterion)
        val_losses.append(val_mse)

        logging.info(
            f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, Val MSE: {val_mse:.6f}")

        # Early Stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0

            # Save best model so far into run_dir
            best_model_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"New best model saved (Val MSE: {best_val_loss:.6f}) at {best_model_path}.")
        else:
            patience_counter += 1
            logging.info(
                f"Val MSE did not improve. Patience: {patience_counter}/{early_stopping_patience}")

        # Checkpointing every N epochs
        if epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch}_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

        # If patience exceeded, stop early
        if patience_counter >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    # --------------------------------------------------------------------------
    # 5) Final Save + Return (with metadata)
    # --------------------------------------------------------------------------
    final_model_path = os.path.join(run_dir, "final_model.pth")

    # Build metadata based on training parameters and results
    metadata = {
        # Last epoch reached (or total epochs if not early stopped)
        "epochs_trained": epoch,
        "lookback_len": lookback_len,
        "forecast_len": forecast_len,
        "batch_size": batch_size,
        "learning_rate": lr,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "run_directory": run_dir,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_kwargs": {
            "d_model": 128,
            "n_heads": 8,
            "d_ff": 256,
            "enc_layers": 3,
            "dec_layers": 2,
            "dropout": 0.1,
            "lookback_len": 24,
            "forecast_len": 12
        }
    }

    # Package the state dictionary and metadata into a single dictionary
    model_package = {
        "state_dict": model.state_dict(),
        "metadata": metadata
    }

    # Save the package to disk
    torch.save(model_package, final_model_path)
    logging.info(
        f"Final model package with metadata saved to {final_model_path}")

    return model, train_losses, val_losses


def evaluate_informer(model, data_loader, device="cpu", criterion=None):
    """
    Evaluate the model on a given data loader. Returns the average MSE if
    criterion is provided, or 0 if not.
    """
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
    mse_avg = mse_sum / max(count, 1)
    return mse_avg


def pretrain_informer(data_source,
                      lookback_len=72,
                      forecast_len=24,
                      epochs=5,
                      batch_size=8,
                      lr=1e-4,
                      device=None,
                      parquet_file=None):
    """
    Pretrain the model using a self-supervised masked prediction task.
    Random segments of the input time series are masked and the model
    is trained to reconstruct those masked parts.
    """
    # (Pseudo-code below; implement according to your data pipeline.)
    # 1. Load your dataset (using GOESParquetDataset, for example) without split.
    # 2. For each batch, randomly mask a percentage of the input values.
    # 3. The target becomes the original values at the masked locations.
    # 4. Use an MSE loss between the reconstruction and the original.
    # 5. Pretrain the encoder (and possibly the decoder) using this task.
    #
    # This pretraining phase can help the model learn general time-series features.
    #
    # Example (pseudocode):
    dataset = GOESParquetDataset(parquet_file=parquet_file,
                                 lookback_len=lookback_len,
                                 forecast_len=forecast_len,
                                 train=True,
                                 train_split=1.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Informer(
        d_model=128, n_heads=8, d_ff=256, enc_layers=3,
        dec_layers=2, dropout=0.1,
        lookback_len=lookback_len, forecast_len=forecast_len
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for x, _ in dataloader:
            x = x.to(device)
            # Create a masked version of x; here we randomly zero out 20% of
            # the entries.
            mask = (torch.rand_like(x) > 0.2).float()
            x_masked = x * mask
            # Target: predict the original x where mask==0.
            target = x
            # Use the model to reconstruct the full sequence.
            # You might use x_masked for both encoder and a dummy decoder
            # input.
            dummy = torch.zeros(x.shape, device=device)
            output = model(x_masked, dummy)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Pretraining Epoch {epoch}: Loss {loss.item()}")
    return model


#######################################################################
# MAIN (OPTIONAL)
#######################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # To use the combined Parquet file (make sure it has been created already):
    # parquet_path = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/models/goes_avg1m_combined.parquet"
    parquet_path = "/Users/antanaszilinskas/Documents/GitHub/masters-project/models/synthetic_flux_data.parquet"
    dev = select_device()

    # Use longer context (72 hours) to predict a shorter future (2 hours)
    train_informer(
        data_source="parquet",
        parquet_file=parquet_path,
        lookback_len=72,    # 72 hours of context
        forecast_len=2,     # 2 hours forecast
        epochs=10,
        batch_size=8,
        lr=1e-6,
        device=dev,
        model_save_path="informer-archive.pth",
        max_train_samples=50000,
        max_val_samples=10000
    )

    # Alternatively, to fall back to netCDF files:
    # data_folder = "/path/to/avg1m/data"
    # train_informer(
    #     data_source="netcdf",
    #     data_dir=data_folder,
    #     lookback_len=72,
    #     forecast_len=24,
    #     epochs=2,
    #     batch_size=16,
    #     lr=1e-4,
    #     device=dev,
    #     max_files=None,
    #     model_save_path="informer-72h-1d.pth"
    # )

    # Optionally, script and save the model for faster deployment.
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, "informer_scripted.pth")
