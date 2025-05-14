import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

# Device config: prefer CUDA, then MPS, then CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Enable cuDNN autotuner when using CUDA (speeds up fixed-size workloads)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ----------------------------------------
# Positional Encoding
# ----------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].type_as(x)

# ----------------------------------------
# Transformer Block
# ----------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.att(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out))

# ----------------------------------------
# Evidential & EVT Losses
# ----------------------------------------
def evidential_nll(y, evid):
    mu, v, a, b = torch.split(evid, 1, dim=-1)
    v = torch.clamp(v, min=1e-3)
    a = torch.clamp(a, min=1.1)
    b = torch.clamp(b, min=1e-3)
    p = torch.clamp(torch.sigmoid(mu), 1e-4, 1 - 1e-4)
    S = b * (1 + v) / a
    eps = 1e-7
    nll = - y * torch.log(p + eps) - (1 - y) * torch.log(1 - p + eps) + 0.5 * torch.log(S + eps)
    return torch.clamp(nll, min=0.0).mean()

def evt_loss(logits, gpd, threshold=2.5):
    try:
        xi, sigma = torch.split(gpd, 1, dim=-1)
        xi = torch.clamp(xi, min=-0.99, max=1.0)
        sigma = torch.clamp(sigma, min=1e-2)
        logits = torch.clamp(logits, max=10.0)
        y = F.relu(logits - threshold)
        mask = (y > 0).squeeze()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        y = y[mask]
        xi = xi[mask]
        sigma = sigma[mask]
        eps = 1e-7
        z = xi * y / (sigma + eps)
        log_term = torch.log1p(torch.clamp(z, min=-0.999, max=1e6))
        term = torch.where(
            torch.abs(xi) < 1e-3,
            y / sigma + torch.log(sigma + eps),
            (1/xi + 1) * log_term + torch.log(sigma + eps)
        )
        reg = 1e-3 * (xi**2).mean() + 1e-3 * (1 / (sigma + eps)).mean()
        return torch.clamp(term.mean(), min=0.0) + reg
    except Exception:
        return torch.tensor(0.0, device=logits.device)

# ----------------------------------------
# Focal Loss
# ----------------------------------------
def focal_bce_loss(logits, targets, gamma):
    targets = targets.view(-1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    mod = (1 - p).pow(gamma) * targets + p.pow(gamma) * (1 - targets)
    return (mod * bce).mean()

# ----------------------------------------
# RET+ Model with attention bottleneck and precursor head
# ----------------------------------------
class RETPlusModel(nn.Module):
    def __init__(self, input_shape, embed_dim=128, num_heads=4, ff_dim=256, num_blocks=6, dropout=0.2):
        super().__init__()
        T, F = input_shape
        self.embedding = nn.Linear(F, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.pos = PositionalEncoding(T, embed_dim)
        self.transformers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        ])

        # Attention bottleneck (learned pooling)
        self.att_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.logits = nn.Linear(128, 1)
        self.nig = nn.Linear(128, 4)
        self.gpd = nn.Linear(128, 2)
        self.precursor_head = nn.Linear(128, 1)  # added precursor head

    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.pos(x)
        for blk in self.transformers:
            x = blk(x)

        attn_weights = self.att_pool(x)  # shape: [B, T, 1]
        x_bottleneck = (x * attn_weights).sum(dim=1)  # shape: [B, embed_dim]

        x = self.head(x_bottleneck)

        logits = self.logits(x)
        nig_raw = self.nig(x)
        mu, logv, loga, logb = torch.split(nig_raw, 1, dim=-1)
        v = F.softplus(logv)
        a = 1 + F.softplus(loga)
        b = F.softplus(logb)
        evid = torch.cat([mu, v, a, b], dim=-1)

        gpd_raw = self.gpd(x)
        xi, log_sigma = torch.split(gpd_raw, 1, dim=-1)
        sigma = F.softplus(log_sigma)
        gpd_out = torch.cat([xi, sigma], dim=-1)

        precursor = self.precursor_head(x)

        return {"logits": logits, "evid": evid, "gpd": gpd_out, "precursor": precursor}

# ----------------------------------------
# Composite Loss
# ----------------------------------------
def composite_loss(y_true, outputs, gamma=0.0, threshold=2.5):
    logits = outputs['logits']
    evid = outputs['evid']
    gpd = outputs['gpd']
    y_true = y_true.view(-1, 1).float()

    fl = focal_bce_loss(logits, y_true, gamma)
    evid_loss = evidential_nll(y_true, evid)
    tail_loss = evt_loss(logits, gpd, threshold)
    total = 0.8 * fl + 0.1 * evid_loss + 0.1 * tail_loss

    return total if not torch.isnan(total) else torch.tensor(0.0, requires_grad=True).to(logits.device)

# ----------------------------------------
# Wrapper
# ----------------------------------------
class RETPlusWrapper:
    def __init__(self, input_shape, early_stopping_patience=10):
        self.model = RETPlusModel(input_shape).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-4
        )
        self.early_stopping_patience = early_stopping_patience
        self.history = {"loss": [], "accuracy": [], "tss": []}
        # Buffers for later interpretability/evaluation artefacts
        self._train_data = None  # tuple (X, y) stored as NumPy arrays

    def train(
        self,
        X_train,
        y_train,
        epochs=100,
        batch_size=512,
        gamma_max=2.0,
        warmup_epochs=50,
        flare_class="M",
        time_window=24,
    ):
        from model_tracking import save_model_with_metadata, get_next_version

        # Convert to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Retain for later evaluation/visualisation
        self._train_data = (X_train, y_train)

        # Build DataLoader for efficient loading
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=4,
        )

        # AMP scaler for mixed-precision
        scaler = GradScaler()

        best_tss = -1e8
        patience = 0
        version = get_next_version(flare_class, time_window)
        best_weights = None
        best_epoch = -1

        for epoch in range(epochs):
            gamma = min(gamma_max, gamma_max * epoch / warmup_epochs)
            epoch_loss = 0.0
            total = 0
            correct = 0
            TP = TN = FP = FN = 0

            for X_batch, y_batch in loader:
                # Move to device
                X_batch = X_batch.to(device, non_blocking=(device.type == "cuda"))
                y_batch = y_batch.to(device, non_blocking=(device.type == "cuda"))

                self.model.train()
                self.optimizer.zero_grad()

                # Mixed-precision forward
                with autocast(device_type=device.type):
                    outputs = self.model(X_batch)
                    loss = composite_loss(y_batch, outputs, gamma=gamma)

                if torch.isnan(loss):
                    continue

                # Scaled backward + step
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Metrics accumulation
                epoch_loss += loss.item()
                preds = (
                    torch.sigmoid(outputs["logits"]) > 0.5
                ).int().squeeze()
                y_true = y_batch.int().squeeze()

                TP += ((preds == 1) & (y_true == 1)).sum().item()
                TN += ((preds == 0) & (y_true == 0)).sum().item()
                FP += ((preds == 1) & (y_true == 0)).sum().item()
                FN += ((preds == 0) & (y_true == 1)).sum().item()
                correct += (preds == y_true).sum().item()
                total += y_batch.size(0)

            # Epoch-level metrics
            avg_loss = epoch_loss / max(1, total)
            accuracy = correct / total if total > 0 else 0.0
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)
            tss = sensitivity + specificity - 1.0

            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(accuracy)
            self.history["tss"].append(tss)

            print(
                f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
                f"acc: {accuracy:.4f} - tss: {tss:.4f} - gamma: {gamma:.2f}"
            )

            # Early stopping check
            if tss > best_tss:
                best_tss = tss
                best_weights = self.model.state_dict()
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. "
                        f"Restoring best model from epoch {best_epoch+1}."
                    )
                    if best_weights is not None:
                        self.model.load_state_dict(best_weights)
                    break

        # Save best model & metadata
        model_dir = self.save(
            version=version,
            flare_class=flare_class,
            time_window=time_window,
            # Provide evaluation data for artefact generation
            X_eval=X_train,
            y_eval=y_train,
        )
        return model_dir

    def predict_proba(self, X):
        self.model.eval()
    
        # If X is already a torch.Tensor (e.g. on CUDA), just move it to the right device
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(device)
        else:
            # Otherwise assume it's a list or NumPy array
            X_np = np.array(X)
            X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
    
        with torch.no_grad():
            logits = self.model(X_tensor)["logits"]
        # Move logits back to CPU before calling .numpy()
        return torch.sigmoid(logits).cpu().numpy()


    def save_weights(self, flare_class, w_dir):
        # Dump raw PyTorch weights
        os.makedirs(w_dir, exist_ok=True)
        weights_path = os.path.join(w_dir, "model_weights.pt")
        torch.save(self.model.state_dict(), weights_path)

    def save(self, version, flare_class, time_window, X_eval=None, y_eval=None):
        from model_tracking import save_model_with_metadata

        metrics = {
            "accuracy": self.history["accuracy"][-1],
            "TSS": self.history["tss"][-1],
        }
        hyperparams = {
            "input_shape": (10, 9),
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 256,
            "num_blocks": 6,
            "dropout": 0.2,
        }

        # --------------------------------------------------------------
        # Optional interpretability & uncertainty artefacts
        # --------------------------------------------------------------
        y_true = y_pred = y_scores = evid_out = evt_scores = None
        att_X = att_y_true = att_y_pred = att_y_score = None
        sample_input = None

        if (X_eval is not None) and (y_eval is not None):
            # Use a small DataLoader to avoid memory blow-up
            eval_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_eval, dtype=torch.float32),
                    torch.tensor(y_eval, dtype=torch.float32),
                ),
                batch_size=512,
                shuffle=False,
            )

            preds_list = []
            probs_list = []
            y_list = []
            evid_list = []
            gpd_list = []

            self.model.eval()
            with torch.no_grad():
                for xb, yb in eval_loader:
                    xb = xb.to(device)
                    out = self.model(xb)
                    prob = torch.sigmoid(out["logits"]).cpu().numpy().flatten()
                    pred = (prob > 0.5).astype(int)
                    preds_list.append(pred)
                    probs_list.append(prob)
                    y_list.append(yb.numpy().flatten())
                    evid_list.append(out["evid"].cpu().numpy())
                    gpd_list.append(out["gpd"].cpu().numpy())

            y_true = np.concatenate(y_list)
            y_scores = np.concatenate(probs_list)
            y_pred = np.concatenate(preds_list)
            evid_out = np.concatenate(evid_list)
            evt_scores = np.concatenate(gpd_list)

            # Attention batch: first 10 samples
            att_X = X_eval[:10]
            att_y_true = y_true[:10]
            att_y_pred = y_pred[:10]
            att_y_score = y_scores[:10]

            sample_input = torch.tensor(att_X[:32], dtype=torch.float32).to(device) if len(X_eval) >= 32 else torch.tensor(X_eval, dtype=torch.float32).to(device)

        model_dir = save_model_with_metadata(
            model=self,
            metrics=metrics,
            hyperparams=hyperparams,
            history=self.history,
            version=version,
            flare_class=flare_class,
            time_window=time_window,
            description="EVEREST model trained on SHARP data with evidential and EVT losses.",
            # Newly added artefacts
            y_true=y_true,
            y_pred=y_pred,
            y_scores=y_scores,
            evt_scores=evt_scores,
            sample_input=sample_input,
            att_X_batch=att_X,
            att_y_true=att_y_true,
            att_y_pred=att_y_pred,
            att_y_score=att_y_score,
            evidential_out=evid_out,
        )
        return model_dir

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))