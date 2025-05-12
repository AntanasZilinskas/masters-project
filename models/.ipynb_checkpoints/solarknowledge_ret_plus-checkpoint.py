# solarknowledge_ret_plus.py (focal loss with gamma annealing + tss-aware early stopping + attention bottleneck + precursor head)

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.early_stopping_patience = early_stopping_patience
        self.history = {"loss": [], "accuracy": [], "tss": []}

    def train(self, X_train, y_train, epochs=100, batch_size=512, gamma_max=2.0, warmup_epochs=50, flare_class="M", time_window=24):
        from model_tracking import save_model_with_metadata, get_next_version
    
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        best_tss = -1e8
        patience = 0
        version = get_next_version(flare_class, time_window)
        best_weights = None
        best_epoch = -1
    
        for epoch in range(epochs):
            gamma = min(gamma_max, gamma_max * epoch / warmup_epochs)
            perm = np.random.permutation(len(X_train))
            epoch_loss = 0.0
            total = 0
            correct = 0
            TP = TN = FP = FN = 0
    
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                X_batch = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_train[idx], dtype=torch.float32).to(device)
    
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = composite_loss(y_batch, outputs, gamma=gamma)
                if torch.isnan(loss):
                    continue
                loss.backward()
                self.optimizer.step()
    
                epoch_loss += loss.item()
                preds = (torch.sigmoid(outputs['logits']) > 0.5).int().squeeze()
                y_true = y_batch.int().squeeze()
    
                TP += ((preds == 1) & (y_true == 1)).sum().item()
                TN += ((preds == 0) & (y_true == 0)).sum().item()
                FP += ((preds == 1) & (y_true == 0)).sum().item()
                FN += ((preds == 0) & (y_true == 1)).sum().item()
                correct += (preds == y_true).sum().item()
                total += y_batch.size(0)
    
            avg_loss = epoch_loss / max(1, (len(X_train) // batch_size))
            accuracy = correct / total if total > 0 else 0.0
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)
            tss = sensitivity + specificity - 1.0
    
            self.history["loss"].append(avg_loss)
            self.history["accuracy"].append(accuracy)
            self.history["tss"].append(tss)
    
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {accuracy:.4f} - tss: {tss:.4f} - gamma: {gamma:.2f}")
    
            if tss > best_tss:
                best_tss = tss
                best_weights = self.model.state_dict()
                best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model from epoch {best_epoch+1}.")
                    if best_weights is not None:
                        self.model.load_state_dict(best_weights)
                    break
    
        # Always save the best model
        self.save(version=version, flare_class=flare_class, time_window=time_window)

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = self.model(X)['logits']
            return torch.sigmoid(logits).cpu().numpy()

    def save(self, version, flare_class, time_window):
        from model_tracking import save_model_with_metadata

        metrics = {
            "accuracy": self.history["accuracy"][-1],
            "TSS": self.history["tss"][-1]
        }
        hyperparams = {
            "input_shape": (10, 9),  # Adjust if input dims are dynamic
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 256,
            "num_blocks": 6,
            "dropout": 0.2,
        }

        save_model_with_metadata(
            model=self,
            metrics=metrics,
            hyperparams=hyperparams,
            history=self.history,
            version=version,
            flare_class=flare_class,
            time_window=time_window,
            description="EVEREST model trained on SHARP data with evidential and EVT losses."
        )

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))