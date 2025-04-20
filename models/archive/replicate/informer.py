import argparse  # For parsing command-line arguments
import datetime
import glob
import logging
import os
import pprint

import matplotlib.pyplot as plt  # For solar cycle analysis plots
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Download the dataset from Google Drive if it doesn't exist locally.
DATA_PATH = "models/replicate/synthetic_flux_data.parquet"
if not os.path.exists(DATA_PATH):
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download the dataset. Please add it to your requirements.txt."
        )

    # The file ID extracted from the provided Google Drive link:
    # https://drive.google.com/file/d/16aedfZ_bGy7_3se2Hce34MYhvcm3r6yC/view?usp=share_link
    file_id = "16aedfZ_bGy7_3se2Hce34MYhvcm3r6yC"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"Dataset not found. Downloading from: {download_url}")
    gdown.download(download_url, DATA_PATH, quiet=False)

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

    def __init__(
        self,
        data_dir,
        lookback_len=72,  # default: 3 days
        forecast_len=24,  # default: 1 day
        step_per_hour=60,
        train=True,
        train_split=0.8,
        max_files=None,
    ):
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
            f"Found {len(all_files)} netCDF files in '{data_dir}' with 'avg1m_g13'"
        )
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"No .nc files matching '*avg1m_g13*.nc' in {data_dir}"
            )

        # ----------------------------------------------------------------------
        # 2) Load flux data (debugging each file)
        # ----------------------------------------------------------------------
        flux_list = []
        for fpath in all_files:
            print(f"[DEBUG] Attempting to open: {fpath}")
            try:
                ds = xr.open_dataset(fpath)
                # Check each possible flux variable
                if "xrsb_flux" in ds.variables:
                    flux_var = "xrsb_flux"
                elif "b_flux" in ds.variables:
                    flux_var = "b_flux"
                elif "a_flux" in ds.variables:
                    flux_var = "a_flux"
                else:
                    ds.close()
                    logging.warning(
                        f"No recognized flux variable in {fpath}, skipping."
                    )
                    continue

                flux_vals = ds[flux_var].values
                ds.close()
                flux_list.append(flux_vals)
                print(
                    f"[DEBUG] Loaded {len(flux_vals)} timesteps from {fpath} (var='{flux_var}')"
                )
            except Exception as err:
                logging.warning(
                    f"Could not load {fpath}, skipping. Error: {err}"
                )
                continue

        if not flux_list:
            raise ValueError(
                "No valid flux data found in selected netCDF files."
            )

        all_flux = np.concatenate(flux_list, axis=0)
        # Do NOT replace NaNs; we want to disregard samples containing null values.
        # Clip values to avoid log1p(x) for x < -1
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
                f"Validation/Testing portion: {len(self.data)} samples"
            )

        # ----------------------------------------------------------------------
        # 4) Build sliding-window indices (+1 fix)
        # ----------------------------------------------------------------------
        max_start = len(self.data) - self.lookback_len - self.forecast_len
        self.indices = []
        for i in range(max_start + 1 if max_start >= 0 else 0):
            self.indices.append(i)

        logging.info(f"Total sliding-window samples: {len(self.indices)}")
        window_length = self.lookback_len + self.forecast_len
        valid_indices = [
            i
            for i in self.indices
            if not np.isnan(self.data[i : i + window_length]).any()
        ]
        logging.info(
            f"Filtered sliding-window samples (without NaN): {len(valid_indices)} out of {len(self.indices)}"
        )
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
        y_seq = self.data[end : end + self.forecast_len]

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)
        return x_tensor, y_tensor


#######################################################################
# 2B) DATASET (Using the precombined Parquet file)
#######################################################################


class GOESParquetDataset(Dataset):
    """
    Loads the precombined Parquet file that contains minute-level 'flux' data
    or aggregated data and builds sliding windows for forecasting.
    """

    def __init__(
        self,
        parquet_file,
        lookback_len=72,
        forecast_len=24,
        train=True,
        train_split=0.8,
        aggregation_freq=None,  # e.g. "M" for monthly aggregation
        is_aggregated=False,
    ):
        super().__init__()
        df = pd.read_parquet(parquet_file)
        if "time" not in df.columns or "flux" not in df.columns:
            raise ValueError(
                "Parquet file must contain 'time' and 'flux' columns."
            )
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)

        if aggregation_freq is not None:
            df.set_index("time", inplace=True)
            df = df.resample(aggregation_freq).mean().reset_index()

        self.data = df["flux"].values.astype(np.float32)
        self.data = np.clip(self.data, a_min=-0.999, a_max=None)
        self.data = np.log1p(self.data)

        if is_aggregated:
            self.lookback_len = lookback_len
            self.forecast_len = forecast_len
        else:
            self.lookback_len = lookback_len * 60  # convert hours to minutes
            self.forecast_len = forecast_len * 60  # convert hours to minutes

        self.train = train
        total_steps = len(self.data)
        split_index = int(total_steps * train_split)
        if train:
            self.data = self.data[:split_index]
            logging.info(
                f"[ParquetDataset] Training portion: {len(self.data)} samples"
            )
        else:
            self.data = self.data[split_index:]
            logging.info(
                f"[ParquetDataset] Validation/Testing portion: {len(self.data)} samples"
            )

        max_start = len(self.data) - self.lookback_len - self.forecast_len
        self.indices = list(range(max_start + 1 if max_start >= 0 else 0))
        logging.info(
            f"[ParquetDataset] Total sliding-window samples: {len(self.indices)}"
        )
        window_length = self.lookback_len + self.forecast_len
        valid_indices = [
            i
            for i in self.indices
            if not np.isnan(self.data[i : i + window_length]).any()
        ]
        logging.info(
            f"[ParquetDataset] Filtered sliding-window samples (without NaN): {len(valid_indices)} out of {len(self.indices)}"
        )
        self.indices = valid_indices

        # Standard-scale the data and store normalization parameters
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
        y_seq = self.data[end : end + self.forecast_len]

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)
        return x_tensor, y_tensor


#######################################################################
# 3) FOURIER ATTENTION (ALTERNATIVE TO STANDARD SELF-ATTENTION)
#######################################################################


class FourierAttention(nn.Module):
    """
    A simple Fourier-based attention module that applies a Fourier transform
    (along the sequence dimension) to mix long-range dependencies.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        xf = torch.fft.rfft(x, dim=0)
        x_out = torch.fft.irfft(xf, n=x.shape[0], dim=0)
        return x_out


#######################################################################
# 3) INFORMER MODEL (MINIMAL VERSION) WITH FOURIER-BASED ATTENTION OPTION
#######################################################################


class TokenEmbedding(nn.Module):
    """
    Embeds the input sequence from (batch_size, 1, seq_len) into (batch_size, d_model, seq_len)
    using a convolution; uses zero padding.
    """

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )

    def forward(self, x):
        out = self.tokenConv(x)
        return out


class PositionalEmbedding(nn.Module):
    """
    Fixed positional encoding for up to max_len positions.
    """

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        Returns positional encodings with the same shape.
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_fourier=False):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.attention = FourierAttention()
        else:
            self.slf_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=False
            )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_fourier:
            x2 = self.attention(x)
        else:
            x2, _ = self.slf_attn(x, x, x)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout(x2)
        x = self.norm2(x)
        return x


class InformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        num_layers,
        dropout=0.1,
        use_fourier=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    use_fourier=use_fourier,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_fourier=False):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            self.attention = FourierAttention()
        else:
            self.slf_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=False
            )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None):
        if self.use_fourier:
            x2 = self.attention(x)
        else:
            x2, _ = self.slf_attn(x, x, x, attn_mask=tgt_mask)
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
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        num_layers,
        dropout=0.1,
        use_fourier=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    use_fourier=use_fourier,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_out, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask)
        return x


class Informer(nn.Module):
    """
    Minimal Informer-like structure for univariate time-series forecasting.
    """

    def __init__(
        self,
        d_model=256,
        n_heads=16,
        d_ff=512,
        enc_layers=6,
        dec_layers=4,
        dropout=0.1,
        lookback_len=72,
        forecast_len=24,
        fourier_attention=False,
    ):
        super().__init__()
        if torch.backends.mps.is_available():
            dropout = 0.0
        self.lookback_len = lookback_len
        self.forecast_len = forecast_len

        self.token_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        self.downsample = nn.Conv1d(
            in_channels=d_model, out_channels=d_model, kernel_size=4, stride=4
        )
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        self.encoder = InformerEncoder(
            d_model,
            n_heads,
            d_ff,
            enc_layers,
            dropout,
            use_fourier=fourier_attention,
        )
        self.decoder = InformerDecoder(
            d_model,
            n_heads,
            d_ff,
            dec_layers,
            dropout,
            use_fourier=fourier_attention,
        )

        self.bridge = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt, use_causal_mask=True, tgt_mask=None):
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape

        # Encoder: embed, downsample, add positional encoding.
        enc_in = src.unsqueeze(1)  # [batch_size, 1, src_len]
        enc_in = self.token_embedding(enc_in)  # [batch_size, d_model, src_len]
        # [batch_size, d_model, reduced_src_len]
        enc_in = self.downsample(enc_in)
        # [reduced_src_len, batch_size, d_model]
        enc_in = enc_in.permute(2, 0, 1)

        # [batch_size, reduced_src_len, d_model]
        enc_in_pe = enc_in.permute(1, 0, 2)
        enc_in_pe = enc_in_pe + self.pos_embedding(enc_in_pe)
        # [reduced_src_len, batch_size, d_model]
        enc_in = enc_in_pe.permute(1, 0, 2)

        # [reduced_src_len, batch_size, d_model]
        enc_out = self.encoder(enc_in)
        # [1, batch_size, d_model]
        bridge_out = self.bridge(enc_out[-1]).unsqueeze(0)

        # Decoder: embed target and add positional encoding.
        dec_in = tgt.unsqueeze(1)  # [batch_size, 1, tgt_len]
        dec_in = self.token_embedding(dec_in)  # [batch_size, d_model, tgt_len]
        dec_in = dec_in.permute(2, 0, 1)  # [tgt_len, batch_size, d_model]

        dec_in_pe = dec_in.permute(1, 0, 2)  # [batch_size, tgt_len, d_model]
        dec_in_pe = dec_in_pe + self.pos_embedding(dec_in_pe)
        dec_in = dec_in_pe.permute(1, 0, 2)  # [tgt_len, batch_size, d_model]

        # Add bridged encoder representation
        dec_in = dec_in + bridge_out

        if use_causal_mask and tgt_mask is None:
            tgt_mask = generate_square_subsequent_mask(tgt_len).to(src.device)

        dec_out = self.decoder(dec_in, enc_out, tgt_mask=tgt_mask)
        dec_out = self.proj(dec_out)
        dec_out = dec_out.permute(1, 0, 2).squeeze(-1)
        return dec_out


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generate a causal mask for the decoder so positions cannot attend
    to subsequent positions. Shape: [sz, sz]
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


#######################################################################
# 4) TRAINING
#######################################################################


def train_informer(
    data_source,
    lookback_len=24,
    forecast_len=12,
    epochs=10,
    batch_size=16,
    lr=1e-6,
    device=None,
    parquet_file=None,
    data_dir=None,
    max_files=None,
    model_save_path="informer-archive.pth",
    early_stopping_patience=3,
    checkpoint_every=2,
    max_train_samples=50000,
    max_val_samples=10000,
    fourier_attention=False,
    d_model=256,
    n_heads=16,
    d_ff=512,
    enc_layers=6,
    dec_layers=4,
    dropout=0.1,
):
    """
    Extended training function that now accepts additional hyperparameters
    for increased model complexity.
    """
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join("model_archives", run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Run directory: {run_dir}")

    if device is None:
        device = select_device()
    logging.info(f"Using device: {device}")

    # Prepare datasets
    if parquet_file is not None:
        logging.info(f"Using Parquet file: {parquet_file}")
        train_dataset = GOESParquetDataset(
            parquet_file=parquet_file,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            train=True,
            train_split=0.8,
        )
        val_dataset = GOESParquetDataset(
            parquet_file=parquet_file,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            train=False,
            train_split=0.8,
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
            max_files=max_files,
        )
        val_dataset = GOESDataset(
            data_dir=data_dir,
            lookback_len=lookback_len,
            forecast_len=forecast_len,
            step_per_hour=60,
            train=False,
            train_split=0.8,
            max_files=max_files,
        )

    if max_train_samples is not None and max_train_samples < len(
        train_dataset.indices
    ):
        train_dataset.indices = train_dataset.indices[:max_train_samples]
        logging.info(
            f"Truncating train dataset to {max_train_samples} samples."
        )

    if max_val_samples is not None and max_val_samples < len(
        val_dataset.indices
    ):
        val_dataset.indices = val_dataset.indices[:max_val_samples]
        logging.info(
            f"Truncating validation dataset to {max_val_samples} samples."
        )

    logging.info(f"Final train samples: {len(train_dataset.indices)}")
    logging.info(f"Final validation samples: {len(val_dataset.indices)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    # Check data statistics on a few samples
    all_x = []
    for idx in train_dataset.indices[:200]:
        x, y = train_dataset[idx]
        all_x.append(x.numpy())
    all_x = np.concatenate(all_x)
    print("Min value in training samples:", np.min(all_x))
    print("Max value in training samples:", np.max(all_x))
    print(
        "Any NaN in training samples?",
        np.isnan(all_x).any(),
        np.isnan(all_x).sum(),
    )
    print("Any Inf in training samples?", np.isinf(all_x).any())
    train_mean = np.nanmean(all_x)
    train_std = np.nanstd(all_x)
    np.save(
        os.path.join(run_dir, "scaling_params.npy"),
        {"mean": train_mean, "std": train_std},
    )

    # Build the model with increased complexity
    model = Informer(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        dropout=dropout,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        fourier_attention=fourier_attention,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch"
        ) as tepoch:
            for batch_idx, (x, y) in enumerate(tepoch):
                x, y = x.to(device), y.to(device)
                batch_size_curr = x.size(0)

                teacher_forcing_ratio = np.exp(-epoch / (epochs / 5))
                if (
                    torch.rand(batch_size_curr).mean().item()
                    < teacher_forcing_ratio
                ):
                    dec_input = y
                else:
                    forecast_steps = y.shape[1]
                    dec_input = x[:, -1:].clone()
                    temperature = 0.7
                    for t in range(forecast_steps - 1):
                        current_length = dec_input.shape[1]
                        dummy = torch.zeros(
                            batch_size_curr, current_length + 1, device=device
                        )
                        dummy[:, :current_length] = dec_input
                        with torch.no_grad():
                            pred_full = model(x, dummy)
                        pred_next = pred_full[:, -1].unsqueeze(1)
                        noise = torch.randn_like(pred_next) * temperature
                        pred_next = pred_next + noise
                        dec_input = torch.cat([dec_input, pred_next], dim=1)

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
        val_mse = evaluate_informer(
            model, val_loader, device=device, criterion=criterion
        )
        val_losses.append(val_mse)
        logging.info(
            f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, Val MSE: {val_mse:.6f}"
        )

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            best_model_path = os.path.join(run_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(
                f"New best model saved (Val MSE: {best_val_loss:.6f}) at {best_model_path}."
            )
        else:
            patience_counter += 1
            logging.info(
                f"Val MSE did not improve. Patience: {patience_counter}/{early_stopping_patience}"
            )

        if epoch % checkpoint_every == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch}_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

        if patience_counter >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

    final_model_path = os.path.join(run_dir, "final_model.pth")
    metadata = {
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
            "d_model": d_model,
            "n_heads": n_heads,
            "d_ff": d_ff,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "dropout": dropout,
            "lookback_len": lookback_len,
            "forecast_len": forecast_len,
            "fourier_attention": fourier_attention,
        },
    }

    model_package = {"state_dict": model.state_dict(), "metadata": metadata}

    torch.save(model_package, final_model_path)
    logging.info(
        f"Final model package with metadata saved to {final_model_path}"
    )
    return model, train_losses, val_losses


def evaluate_informer(model, data_loader, device="cpu", criterion=None):
    """
    Evaluate the model on a given data loader. Returns the average MSE.
    """
    model.eval()
    mse_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            dummy = torch.zeros_like(y).to(device)
            pred = model(x, dummy)
            if criterion is not None:
                mse_sum += criterion(pred, y).item()
            count += 1
    return mse_sum / max(count, 1)


#######################################################################
# SOLAR CYCLE ANALYSIS
#######################################################################


def analyze_solar_cycle_patterns(model, data_loader, device="cpu"):
    """
    Preliminary analysis of solar cycle patterns using FFT-based methods.
    """
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            dummy = torch.zeros_like(y).to(device)
            pred = model(x, dummy)
            predictions.append(pred.cpu().numpy())
            ground_truth.append(y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(predictions, label="Predictions")
    plt.plot(ground_truth, label="Ground Truth", alpha=0.7)
    plt.title("Forecast vs Ground Truth")
    plt.legend()
    plt.show()

    fft_vals = np.abs(np.fft.fft(predictions.flatten()))
    freq = np.fft.fftfreq(len(fft_vals), d=1)
    plt.figure(figsize=(12, 5))
    plt.plot(freq[: len(freq) // 2], fft_vals[: len(fft_vals) // 2])
    plt.title("FFT Spectrum of Predictions")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()

    logging.info(
        "Solar cycle analysis complete. Review the FFT spectrum for dominant periodicities."
    )


#######################################################################
# PRETRAINING
#######################################################################


def pretrain_informer(
    data_source,
    lookback_len=72,
    forecast_len=24,
    epochs=5,
    batch_size=16,
    lr=1e-4,
    device=None,
    parquet_file=None,
):
    """
    Pretrain the model using a self-supervised masked prediction task.
    """
    dataset = GOESParquetDataset(
        parquet_file=parquet_file,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        train=True,
        train_split=1.0,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Informer(
        d_model=256,
        n_heads=16,
        d_ff=512,
        enc_layers=6,
        dec_layers=4,
        dropout=0.1,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for x, _ in dataloader:
            x = x.to(device)
            mask = (torch.rand_like(x) > 0.2).float()
            x_masked = x * mask
            target = x
            dummy = torch.zeros(x.shape, device=device)
            output = model(x_masked, dummy)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Pretraining Epoch {epoch}: Loss {loss.item()}")
    return model


#######################################################################
# MAIN
#######################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Informer model with increased complexity on cloud GPUs."
    )
    parser.add_argument(
        "--train", action="store_true", help="If set, run training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, help="Learning rate."
    )
    parser.add_argument(
        "--lookback_len",
        type=int,
        default=72,
        help="Lookback length (in hours or aggregated units).",
    )
    parser.add_argument(
        "--forecast_len",
        type=int,
        default=2,
        help="Forecast length (in hours).",
    )
    parser.add_argument(
        "--parquet_file",
        type=str,
        default=None,
        help="Path to the Parquet file for training.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory for netCDF files (if parquet_file is not provided).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=50000,
        help="Max training samples to use.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=10000,
        help="Max validation samples to use.",
    )
    parser.add_argument(
        "--fourier_attention",
        action="store_true",
        help="Enable Fourier-based attention.",
    )
    # New hyperparameters for model complexity
    parser.add_argument(
        "--d_model", type=int, default=256, help="Model dimensionality."
    )
    parser.add_argument(
        "--n_heads", type=int, default=16, help="Number of attention heads."
    )
    parser.add_argument(
        "--d_ff", type=int, default=512, help="Feed-forward network dimension."
    )
    parser.add_argument(
        "--enc_layers", type=int, default=6, help="Number of encoder layers."
    )
    parser.add_argument(
        "--dec_layers", type=int, default=4, help="Number of decoder layers."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="informer-archive.pth",
        help="Path to save the final model package.",
    )
    args = parser.parse_args()

    if args.train:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        device = select_device()
        logging.info(f"Using device: {device}")

        model, train_losses, val_losses = train_informer(
            data_source="parquet"
            if args.parquet_file is not None
            else "netcdf",
            parquet_file=args.parquet_file,
            data_dir=args.data_dir,
            lookback_len=args.lookback_len,
            forecast_len=args.forecast_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            fourier_attention=args.fourier_attention,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            dropout=args.dropout,
            model_save_path=args.model_save_path,
        )

        # Optionally, script and save the model for deployment.
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, "informer_scripted.pth")
        logging.info("Scripted model saved as 'informer_scripted.pth'.")
