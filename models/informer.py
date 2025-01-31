import os
import glob
import logging
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------------
# Set up basic logging configuration
# Feel free to change the log level (DEBUG, INFO, WARNING, ERROR) as needed.
# You can also customize the format to suit your preferences.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

############################################################
# 1) DATASET & DATA LOADING
############################################################

class GOESDataset(Dataset):
    """
    Loads GOES netCDF files containing "avg1m" in the filename
    (e.g., sci_xrsf-l2-avg1m_g16_d20180908_v2-2-0.nc), merges them
    chronologically, and creates sliding windows (lookback -> forecast).
    """

    def __init__(self,
                 data_dir,
                 lookback_len=24,     # hours
                 forecast_len=24,     # hours
                 step_per_hour=60,    # 1-min data => 60 steps/hour
                 train=True,
                 train_split=0.8,
                 max_files=None):
        super().__init__()
        self.lookback_len = lookback_len * step_per_hour
        self.forecast_len = forecast_len * step_per_hour
        self.train = train

        logging.debug(
            f"GOESDataset init: data_dir={data_dir}, lookback_len={lookback_len}, "
            f"forecast_len={forecast_len}, step_per_hour={step_per_hour}, train={train}"
        )

        # ------------------------------------------------------------------------------
        # Use the EXACT method for fetching files with 'avg1m' in the name (no ellipses).
        # ------------------------------------------------------------------------------
        pattern = os.path.join(data_dir, "*avg1m*.nc")
        logging.debug(f"DEBUG: Looking for files with pattern: {pattern}")

        all_files = sorted(glob.glob(pattern))
        logging.debug(f"DEBUG: All matching files found: {all_files}")

        if max_files is not None:
            all_files = all_files[:max_files]

        logging.info(f"Found {len(all_files)} netCDF files containing 'avg1m' in {data_dir}")

        if len(all_files) == 0:
            # If no files found, raise a FileNotFoundError
            raise FileNotFoundError(f"No .nc files matching '*avg1m*.nc' found in {data_dir}")

        # ------------------------------------------------------------------------------
        # 2) Load flux data from each file
        # ------------------------------------------------------------------------------
        flux_list = []
        for fpath in all_files:
            try:
                logging.debug(f"Opening file: {fpath}")
                ds = xr.open_dataset(fpath)
                # Choose a flux variable if it exists
                if 'xrsb_flux' in ds.variables:
                    flux_var = 'xrsb_flux'
                elif 'b_flux' in ds.variables:
                    flux_var = 'b_flux'
                elif 'a_flux' in ds.variables:
                    flux_var = 'a_flux'
                else:
                    # If none of these variables are present, skip the file
                    ds.close()
                    logging.warning(f"No recognized flux variable in: {fpath}, skipping.")
                    continue

                flux_vals = ds[flux_var].values
                ds.close()
                flux_list.append(flux_vals)
                logging.debug(f"Loaded {len(flux_vals)} timesteps from {fpath}")

            except Exception as err:
                logging.warning(f"Could not load {fpath}, skipping. Error: {err}")
                continue

        if len(flux_list) == 0:
            raise ValueError("No valid flux data found among the selected netCDF files.")

        # Concatenate all flux arrays into a single time-series
        all_flux = np.concatenate(flux_list, axis=0)

        # 3) Fill NaNs and apply small transform if desired
        all_flux = np.nan_to_num(all_flux, nan=1e-9)
        self.data = np.log1p(all_flux)  # For instance, log-transform

        # 4) Train/test split
        N = len(self.data)
        split_index = int(N * train_split)
        if train:
            self.data = self.data[:split_index]
            logging.info(f"Training portion: {len(self.data)} samples")
        else:
            self.data = self.data[split_index:]
            logging.info(f"Validation/Testing portion: {len(self.data)} samples")

        # 5) Build sliding window indices
        self.indices = []
        max_start = len(self.data) - self.lookback_len - self.forecast_len
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

        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.float32)
        return x_tensor, y_tensor


############################################################
# 2) INFORMER MODEL IMPLEMENTATION (Simplified)
############################################################

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x shape: [Batch, c_in, seq_len]
        out = self.tokenConv(x)
        out = self.bn(out)
        return out


class PositionalEmbedding(nn.Module):
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, seq_len, d_model]
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class EncoderLayer(nn.Module):
    """
    A simplified self-attention encoder layer (akin to ProbSparse idea).
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout
        )
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # x shape: [seq_len, Batch, d_model]
        x2, _ = self.slf_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
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
        attn_mask = None
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return x


class DecoderLayer(nn.Module):
    """
    Basic naive decoder layer for an attention-based architecture.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        # x shape: [tgt_seq_len, Batch, d_model]
        # enc_out shape: [src_seq_len, Batch, d_model]

        # Self-attention
        x2, _ = self.slf_attn(x, x, x)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        # Cross-attention
        x2, _ = self.cross_attn(x, enc_out, enc_out)
        x = x + self.dropout(x2)
        x = self.norm2(x)

        # Feed-forward
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
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
    
    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


class Informer(nn.Module):
    """
    A minimal Informer structure:
      - 1D conv-based token embedding -> positional embedding -> stacked encoder
        -> a small decoder -> final projection
    """
    def __init__(self,
                 d_model=64,
                 n_heads=4,
                 d_ff=128,
                 enc_layers=2,
                 dec_layers=1,
                 dropout=0.1,
                 lookback_len=24,
                 forecast_len=24):
        super().__init__()
        self.lookback_len = lookback_len
        self.forecast_len = forecast_len

        # For univariate flux data, c_in=1
        self.token_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        # Encoders/decoders
        self.encoder = InformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=enc_layers,
            dropout=dropout
        )
        self.decoder = InformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=dec_layers,
            dropout=dropout
        )

        # Output (project to 1 channel)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        """
        src shape: [batch, lookback_len]
        tgt shape: [batch, forecast_len]
        """
        logging.debug(f"Informer forward called with src.shape={src.shape}, tgt.shape={tgt.shape}")

        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape

        # Prepare for encoder
        enc_in = src.unsqueeze(1)              # -> [batch, 1, src_len]
        enc_in = self.token_embedding(enc_in)  # -> [batch, d_model, src_len]
        enc_in = enc_in.permute(2, 0, 1)       # -> [src_len, batch, d_model]
        enc_in = enc_in + self.pos_embedding(enc_in.permute(1, 0, 2)).permute(1, 0, 2)
        enc_out = self.encoder(enc_in)         # -> [src_len, batch, d_model]
        logging.debug(f"Encoder output shape: {enc_out.shape}")

        # Prepare for decoder (naive approach: zero future)
        dec_in = torch.zeros(batch_size, 1, tgt_len).to(src.device)  # [batch, 1, tgt_len]
        dec_in = self.token_embedding(dec_in)                        # -> [batch, d_model, tgt_len]
        dec_in = dec_in.permute(2, 0, 1)                              # -> [tgt_len, batch, d_model]
        dec_in = dec_in + self.pos_embedding(dec_in.permute(1, 0, 2)).permute(1, 0, 2)

        dec_out = self.decoder(dec_in, enc_out)  # -> [tgt_len, batch, d_model]
        logging.debug(f"Decoder output shape: {dec_out.shape}")

        # Project to univariate
        dec_out = self.proj(dec_out)               # -> [tgt_len, batch, 1]
        dec_out = dec_out.permute(1, 0, 2).squeeze(-1)  # -> [batch, tgt_len]
        logging.debug(f"Final projection output shape: {dec_out.shape}")

        return dec_out


############################################################
# 3) TRAINING / INFERENCE EXAMPLE
############################################################

def select_device():
    """
    Prefers MPS (Metal) on Apple Silicon, otherwise CUDA if available,
    else CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_informer(
    data_dir="/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024",
    lookback_len=24,       # 24-hour history
    forecast_len=24,       # 24-hour forecast
    epochs=2,
    batch_size=16,
    lr=1e-4,
    device='cpu',
    max_files=None,
    model_save_path="informer-24h-new_test_split.pth"
):
    """
    data_dir       : folder with netCDF files (contains 'avg1m' in filenames)
    lookback_len   : hours of history
    forecast_len   : hours to forecast
    epochs         : full training epochs
    batch_size     : batch size
    lr             : learning rate
    device         : CPU, CUDA, or MPS
    max_files      : limit how many daily netCDF files to load (for debugging)
    model_save_path: path to save model weights
    """

    logging.info("Starting train_informer with 24h->24h configuration.")
    logging.info(f"Params: lookback_len={lookback_len}, forecast_len={forecast_len}, "
                 f"epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}, "
                 f"max_files={max_files}, model_save_path={model_save_path}")

    # 1) Create dataset & loaders
    train_dataset = GOESDataset(
        data_dir=data_dir,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        step_per_hour=60,     # 1-min resolution
        train=True,
        train_split=0.8,
        max_files=max_files
    )

    test_dataset = GOESDataset(
        data_dir=data_dir,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        step_per_hour=60,
        train=False,
        train_split=0.8,
        max_files=max_files
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info(f"Train loader size (batches): {len(train_loader)}, "
                 f"Test loader size (batches): {len(test_loader)}")

    # 2) Initialize the Informer model
    model = Informer(
        d_model=64,
        n_heads=4,
        d_ff=128,
        enc_layers=2,
        dec_layers=1,
        dropout=0.1,
        lookback_len=lookback_len,
        forecast_len=forecast_len
    )
    model.to(device)

    logging.info("Model initialized.")
    logging.debug(model)

    # 3) Define loss + optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4) Training loop
    model.train()
    logging.info("Beginning training...")
    for ep in range(epochs):
        total_loss = 0
        batch_count = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            pred = model(x, y)
            loss = criterion(pred, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                logging.debug(f"Epoch[{ep+1}/{epochs}], Batch[{batch_idx+1}/{len(train_loader)}], "
                              f"Loss={loss.item():.6f}")

        # End-of-epoch
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        logging.info(f"Epoch [{ep+1}/{epochs}] - Train Loss: {avg_loss:.6f}")

    # 5) Evaluate on test set
    logging.info("Evaluating on test set...")
    model.eval()
    test_mse = 0.0
    count = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x, y)
            loss = criterion(pred, y)
            test_mse += loss.item()
            count += 1

    test_mse_avg = test_mse / max(count, 1)
    logging.info(f"Test MSE: {test_mse_avg:.6f}")

    # 6) Save the model weights
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model weights saved to {model_save_path}")

    return model


#######################################################################
# MAIN ENTRY (OPTIONAL)
#######################################################################
if __name__ == "__main__":
    # Use the full path without ellipses:
    DATA_DIR = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"

    device = select_device()
    logging.info(f"Using device: {device}")

    # Start small: load 5 daily files, each ~1440 steps (24h)
    trained_model = train_informer(
        data_dir=DATA_DIR,
        lookback_len=24,
        forecast_len=24,
        epochs=2,
        batch_size=16,
        lr=1e-4,
        device=device,
        max_files=5,  # Only load 5 daily files for debugging
        model_save_path="informer-24h-new_test_split.pth"
    )

    # Quick forward pass check with random input
    trained_model.eval()
    with torch.no_grad():
        # For 24-hour windows: input shape => (batch=1, seq_len=24)
        example_x = torch.randn(1, 24, dtype=torch.float32).to(device)
        dummy_y   = torch.zeros(1, 24, dtype=torch.float32).to(device)
        pred_future = trained_model(example_x, dummy_y)
        logging.info(f"Predicted shape: {pred_future.shape}") 