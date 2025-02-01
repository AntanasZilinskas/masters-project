import os
import glob
import logging
import pprint
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
# 2) DATASET
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

        logging.info(f"Found {len(all_files)} netCDF files in '{data_dir}' with 'avg1m_g13'")
        if len(all_files) == 0:
            raise FileNotFoundError(f"No .nc files matching '*avg1m_g13*.nc' in {data_dir}")

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
                    logging.warning(f"No recognized flux variable in {fpath}, skipping.")
                    continue

                flux_vals = ds[flux_var].values
                ds.close()
                flux_list.append(flux_vals)
                print(f"[DEBUG] Loaded {len(flux_vals)} timesteps from {fpath} (var='{flux_var}')")
            except Exception as err:
                logging.warning(f"Could not load {fpath}, skipping. Error: {err}")
                continue

        if not flux_list:
            raise ValueError("No valid flux data found in selected netCDF files.")

        all_flux = np.concatenate(flux_list, axis=0)
        all_flux = np.nan_to_num(all_flux, nan=1e-9)
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
            logging.info(f"Validation/Testing portion: {len(self.data)} samples")

        # ----------------------------------------------------------------------
        # 4) Build sliding-window indices (+1 fix)
        # ----------------------------------------------------------------------
        max_start = len(self.data) - self.lookback_len - self.forecast_len
        self.indices = []
        for i in range(max_start + 1 if max_start >= 0 else 0):
            self.indices.append(i)

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

#######################################################################
# 3) INFORMER MODEL (MINIMAL VERSION)
#######################################################################
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
        out = self.tokenConv(x)
        out = self.bn(out)
        return out

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.slf_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x2, _ = self.slf_attn(x, x, x)
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
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
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
        x2, _ = self.slf_attn(x, x, x)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2, _ = self.cross_attn(x, enc_out, enc_out)
        x = x + self.dropout(x2)
        x = self.norm2(x)

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
    Minimal Informer structure for univariate time-series.
    """
    def __init__(self,
                 d_model=64,
                 n_heads=4,
                 d_ff=128,
                 enc_layers=2,
                 dec_layers=1,
                 dropout=0.1,
                 lookback_len=72,
                 forecast_len=24):
        super().__init__()
        self.lookback_len = lookback_len
        self.forecast_len = forecast_len

        self.token_embedding = TokenEmbedding(c_in=1, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.encoder = InformerEncoder(d_model, n_heads, d_ff, enc_layers, dropout)
        self.decoder = InformerDecoder(d_model, n_heads, d_ff, dec_layers, dropout)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        """
        src: [batch_size, lookback_len]
        tgt: [batch_size, forecast_len]
        """
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape

        # Encoder input
        enc_in = src.unsqueeze(1)
        enc_in = self.token_embedding(enc_in)
        enc_in = enc_in.permute(2, 0, 1)
        enc_in = enc_in + self.pos_embedding(enc_in.permute(1, 0, 2)).permute(1, 0, 2)
        enc_out = self.encoder(enc_in)

        # Decoder input (zero placeholder for future sequence)
        dec_in = torch.zeros(batch_size, 1, tgt_len).to(src.device)
        dec_in = self.token_embedding(dec_in)
        dec_in = dec_in.permute(2, 0, 1)
        dec_in = dec_in + self.pos_embedding(dec_in.permute(1, 0, 2)).permute(1, 0, 2)
        dec_out = self.decoder(dec_in, enc_out)

        # Project to scalar
        dec_out = self.proj(dec_out)
        dec_out = dec_out.permute(1, 0, 2).squeeze(-1)
        return dec_out

#######################################################################
# 4) TRAINING
#######################################################################
def train_informer(data_dir,
                   lookback_len=72,     # 3 days
                   forecast_len=24,     # 1 day
                   epochs=5,
                   batch_size=16,
                   lr=1e-4,
                   device=None,
                   max_files=None,
                   model_save_path="informer-72h-1d.pth"):
    if device is None:
        device = select_device()
    logging.info(f"Using device: {device}")

    # Prepare datasets
    train_dataset = GOESDataset(
        data_dir=data_dir,
        lookback_len=lookback_len,
        forecast_len=forecast_len,
        step_per_hour=60,
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
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

    # Training loop
    for epoch in range(1, epochs + 1):
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
        logging.info(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_loss:.6f}")

        # Log how many days were in the training set
        train_hours = len(train_dataset.data)
        train_days = train_hours / (24.0 * 60)
        logging.info(f"Trained so far on ~{train_days:.2f} days of data")

        # Evaluate on test set
        test_mse = evaluate_informer(model, test_loader, device=device, criterion=criterion)
        logging.info(f"Test MSE after epoch {epoch}: {test_mse:.6f}")

    # Save model
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model weights saved to {model_save_path}")

    return model

def evaluate_informer(model, data_loader, device='cpu', criterion=None):
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

#######################################################################
# MAIN (OPTIONAL)
#######################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    data_folder = "/Users/antanaszilinskas/Desktop/Imperial College London/D2P/Coursework/masters-project/data/GOES/data/avg1m_2010_to_2024"
    dev = select_device()
    train_informer(
        data_dir=data_folder,
        lookback_len=72,       # 3 days
        forecast_len=24,       # 1 day
        epochs=2,
        batch_size=16,
        lr=1e-4,
        device=dev,
        max_files=None,
        model_save_path="informer-72h-1d.pth"
    ) 